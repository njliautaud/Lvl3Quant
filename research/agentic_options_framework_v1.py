#!/usr/bin/env python3
"""
Agentic Options Framework v1
==============================
Translates contrarian stock-picking signals into option trade recommendations
for a small agentic Robinhood account (Level 2: buy calls/puts, spreads).

Signal types from validated contrarian engine:
  1. Skewness Premium (Sharpe 1.02, hold 10d)
  2. Post-Earnings Drift Contrarian (Sharpe 2.11, hold 21d)
  3. Smart Money Accumulation (Sharpe 1.61, hold 10d)
  4. Price-Volume Divergence (Sharpe 1.23, hold 10d)
  5. Volatility Crush Reversal (Sharpe 0.96, hold 21d)

Usage:
    from agentic_options_framework_v1 import OptionsFramework
    fw = OptionsFramework(account_value=677.0)
    rec = fw.evaluate_signals("AAPL", signals=[...])
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from enum import Enum
from typing import Optional


# ── Signal Definitions ───────────────────────────────────────────────────────

class SignalType(str, Enum):
    OVERSOLD_BOUNCE = "oversold_bounce"            # BACKTEST VALIDATED (p=0.01)
    POST_EARNINGS_BOUNCE = "post_earnings_bounce"  # BACKTEST VALIDATED (all 5 gates)
    SKEWNESS_PREMIUM = "skewness_premium"
    POST_EARNINGS_DRIFT = "post_earnings_drift"
    SMART_MONEY = "smart_money"
    PRICE_VOLUME_DIVERGENCE = "price_volume_divergence"
    VOL_CRUSH_REVERSAL = "vol_crush_reversal"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    SECTOR_MOMENTUM = "sector_momentum"


# Validated signal characteristics from paper trading engine + backtests
SIGNAL_PROFILES: dict[str, dict] = {
    # ── OVERSOLD BOUNCE — THE ONLY BACKTEST-VALIDATED SIGNAL (p=0.01) ──
    # Backtest: 1,064 trades, 2015-2026, 20 S&P 500 stocks
    # Best variant: 10% weekly drop, 5-day hold. 58.5% WR, PF 1.71
    # Bear market performance BETTER: 89% WR, +5.5% avg return
    SignalType.OVERSOLD_BOUNCE: {
        "base_win_rate": 0.585,           # from backtest (58.5%)
        "expected_return_pct": 1.75,      # avg stock return
        "hold_days": 5,                   # backtest-optimal
        "sharpe": 0.197,                  # per-trade Sharpe (low but consistent)
        "direction": "long",             # buy the bounce
        "base_confidence": 0.75,         # HIGH — only validated signal
        "permutation_p": 0.01,           # PASSES statistical test
        "regime_gap": 0.32,              # works in ALL regimes
        "backtest_validated": True,
    },
    # ── POST-EARNINGS BOUNCE — SECOND VALIDATED SIGNAL (all 5 gates pass) ──
    # Backtest: 51 trades, 2018-2026, 20 S&P 500 stocks
    # 8%+ drop in 2 days after earnings, buy day 3, hold 10 days
    # Regime-agnostic (gap 0.33), sub-period stable, survives outlier removal
    SignalType.POST_EARNINGS_BOUNCE: {
        "base_win_rate": 0.627,           # from backtest (62.7%)
        "expected_return_pct": 1.58,      # avg stock return
        "hold_days": 10,                  # backtest-optimal (10d hold passes all gates)
        "sharpe": 1.50,
        "sortino": 2.38,
        "direction": "long",             # buy the post-earnings bounce
        "base_confidence": 0.70,         # HIGH — all 5 adversarial gates pass
        "permutation_p": 0.02,
        "regime_gap": 0.33,
        "backtest_validated": True,
    },
    SignalType.SKEWNESS_PREMIUM: {
        "base_win_rate": 0.57,
        "expected_return_pct": 1.2,   # avg stock return when signal works
        "hold_days": 10,
        "sharpe": 1.02,
        "direction": "long",         # contrarian = buy after drop
        "base_confidence": 0.55,
    },
    SignalType.POST_EARNINGS_DRIFT: {
        "base_win_rate": 0.64,
        "expected_return_pct": 2.8,
        "hold_days": 21,
        "sharpe": 2.11,
        "direction": "long",
        "base_confidence": 0.70,
    },
    SignalType.SMART_MONEY: {
        "base_win_rate": 0.61,
        "expected_return_pct": 1.8,
        "hold_days": 10,
        "sharpe": 1.61,
        "direction": "long",
        "base_confidence": 0.65,
    },
    SignalType.PRICE_VOLUME_DIVERGENCE: {
        "base_win_rate": 0.58,
        "expected_return_pct": 1.5,
        "hold_days": 10,
        "sharpe": 1.23,
        "direction": "long",
        "base_confidence": 0.58,
    },
    SignalType.VOL_CRUSH_REVERSAL: {
        "base_win_rate": 0.54,
        "expected_return_pct": 2.0,
        "hold_days": 21,
        "sharpe": 0.96,
        "direction": "long",
        "base_confidence": 0.50,
    },
    SignalType.MOMENTUM_EXHAUSTION: {
        "base_win_rate": 0.56,
        "expected_return_pct": 1.4,
        "hold_days": 7,
        "sharpe": 1.10,
        "direction": "long",         # buy oversold bounce
        "base_confidence": 0.52,
    },
    SignalType.SECTOR_MOMENTUM: {
        "base_win_rate": 0.55,
        "expected_return_pct": 0.9,
        "hold_days": 14,
        "sharpe": 0.85,
        "direction": "long",
        "base_confidence": 0.48,
    },
}


@dataclass
class Signal:
    """A single contrarian signal firing on a stock."""
    signal_type: SignalType
    confidence: float           # 0-1, signal-specific confidence
    indicators: dict = field(default_factory=dict)  # RSI, MFI, OBV, etc.

    @property
    def profile(self) -> dict:
        return SIGNAL_PROFILES[self.signal_type]


@dataclass
class MacroContext:
    """Macro/sector context for confluence scoring."""
    sector_momentum_aligned: bool = False   # sector ETF trending with signal
    cta_flow_supportive: bool = False       # CTA positioning supportive
    breadth_positive: bool = False          # market breadth expanding
    vix_regime: str = "normal"              # "low", "normal", "elevated", "crisis"
    risk_appetite: str = "neutral"          # "risk_on", "neutral", "risk_off"


@dataclass
class TradeRecommendation:
    """Structured output for a trade decision."""
    action: str                    # BUY_CALL, BUY_PUT, BULL_CALL_SPREAD, SKIP
    ticker: str
    strike: Optional[float] = None
    expiry: Optional[str] = None
    contracts: int = 0
    max_cost: float = 0.0
    confidence: str = "LOW"        # HIGH, MEDIUM, LOW
    confluence_score: float = 0.0
    signals_firing: list[str] = field(default_factory=list)
    expected_stock_move_pct: float = 0.0
    expected_option_return_pct: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    time_stop_dte: int = 3
    reasoning: str = ""
    risk_pct_of_account: float = 0.0
    kelly_fraction: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.action == "SKIP":
            return (
                f"SKIP {self.ticker} | Confluence={self.confluence_score:.2f} | "
                f"Reason: {self.reasoning}"
            )
        return (
            f"{self.action} {self.ticker} ${self.strike} exp {self.expiry} | "
            f"{self.contracts} contract(s) @ max ${self.max_cost:.0f} | "
            f"Confluence={self.confluence_score:.2f} ({self.confidence}) | "
            f"Expected stock +{self.expected_stock_move_pct:.1f}% -> "
            f"option +{self.expected_option_return_pct:.0f}% | "
            f"SL=${self.stop_loss_price:.0f} TP=${self.take_profit_price:.0f} | "
            f"Risk={self.risk_pct_of_account:.0f}% of account"
        )


# ── Risk Rules (hard-coded, non-negotiable) ──────────────────────────────────

@dataclass
class RiskRules:
    """Level 2 Robinhood account risk constraints."""
    max_single_trade_pct: float = 0.50       # 50% of account max
    max_position_pct: float = 0.25           # 25% of account per position
    max_concurrent_positions: int = 3
    stop_loss_pct: float = 0.50              # exit if option loses 50%
    take_profit_pct: float = 1.00            # exit if option gains 100% (2x)
    time_stop_dte: int = 3                   # exit within 3 DTE
    min_dte_at_entry: int = 7                # never buy < 7 DTE
    max_dte: int = 45                        # never buy > 45 DTE
    min_option_price: float = 0.10           # no penny options
    max_option_price_pct: float = 0.25       # single option cost < 25% acct


# ── Core Framework ───────────────────────────────────────────────────────────

class OptionsFramework:
    """
    Translates contrarian stock signals into option trade recommendations.

    Parameters
    ----------
    account_value : float
        Current account equity (default $677).
    current_positions : int
        Number of currently open positions.
    risk_rules : RiskRules
        Risk constraints. Defaults are conservative for small account.
    """

    def __init__(
        self,
        account_value: float = 677.0,
        current_positions: int = 0,
        risk_rules: Optional[RiskRules] = None,
    ):
        self.account_value = account_value
        self.current_positions = current_positions
        self.rules = risk_rules or RiskRules()

    # ── Confluence Scoring ───────────────────────────────────────────────

    def compute_confluence(
        self,
        signals: list[Signal],
        macro: Optional[MacroContext] = None,
    ) -> tuple[float, str]:
        """
        Compute confluence score from multiple signals + macro context.

        Returns (score, confidence_label) where:
          score in [0, 1] — higher = more signals agreeing
          confidence_label: "HIGH" (>=0.70), "MEDIUM" (>=0.45), "LOW" (<0.45)
        """
        if not signals:
            return 0.0, "LOW"

        macro = macro or MacroContext()

        # Base: weighted average of signal confidences
        # Weight by signal Sharpe ratio (better signals count more)
        total_weight = 0.0
        weighted_conf = 0.0
        for sig in signals:
            sharpe = sig.profile["sharpe"]
            weight = sharpe
            weighted_conf += sig.confidence * weight
            total_weight += weight

        base_score = weighted_conf / total_weight if total_weight > 0 else 0.0

        # Multi-signal bonus: reward confluence of independent signals
        n_signals = len(signals)
        if n_signals == 1:
            multi_bonus = 0.0
        elif n_signals == 2:
            multi_bonus = 0.12
        elif n_signals == 3:
            multi_bonus = 0.20
        else:
            multi_bonus = 0.25  # diminishing returns past 3

        # Macro alignment bonus (each adds a small bump)
        macro_bonus = 0.0
        if macro.sector_momentum_aligned:
            macro_bonus += 0.05
        if macro.cta_flow_supportive:
            macro_bonus += 0.04
        if macro.breadth_positive:
            macro_bonus += 0.03

        # VIX regime adjustment
        vix_adj = {
            "low": 0.02,       # low vol = easier for long calls
            "normal": 0.0,
            "elevated": -0.05, # premium expensive, harder to profit
            "crisis": -0.15,   # don't buy options in panic
        }.get(macro.vix_regime, 0.0)

        # Risk appetite adjustment
        risk_adj = {
            "risk_on": 0.03,
            "neutral": 0.0,
            "risk_off": -0.08,
        }.get(macro.risk_appetite, 0.0)

        score = min(1.0, max(0.0, base_score + multi_bonus + macro_bonus + vix_adj + risk_adj))

        if score >= 0.70:
            label = "HIGH"
        elif score >= 0.45:
            label = "MEDIUM"
        else:
            label = "LOW"

        return score, label

    # ── Kelly Criterion for Options ──────────────────────────────────────

    @staticmethod
    def kelly_fraction(
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
    ) -> float:
        """
        Half-Kelly for option sizing. Returns fraction of bankroll to risk.

        For options:
          avg_win = expected option return when trade works
          avg_loss = expected loss (usually ~50% due to stop loss)
        """
        if avg_loss_pct == 0 or avg_win_pct <= 0:
            return 0.0

        b = avg_win_pct / avg_loss_pct  # odds ratio
        p = win_rate
        q = 1 - p

        kelly = (b * p - q) / b
        # Half-Kelly for safety (especially on small account)
        half_kelly = max(0.0, kelly / 2)
        return min(half_kelly, 0.25)  # cap at 25% regardless

    # ── Strike Selection ─────────────────────────────────────────────────

    @staticmethod
    def select_strike(
        current_price: float,
        expected_move_pct: float,
        confidence_label: str,
        direction: str = "long",
    ) -> float:
        """
        Select strike price based on confidence level.

        HIGH confidence -> slightly ITM (higher delta, more expensive but higher P(profit))
        MEDIUM confidence -> ATM (balanced delta/cost)
        LOW confidence -> slightly OTM (cheaper, lower P(profit) but better risk/reward)
        """
        if direction == "long":
            # Buying calls for bullish contrarian signals
            offsets = {
                "HIGH": -0.02,    # 2% ITM
                "MEDIUM": 0.0,    # ATM
                "LOW": 0.02,      # 2% OTM
            }
        else:
            # Buying puts (rare for contrarian, but supported)
            offsets = {
                "HIGH": 0.02,     # 2% ITM for puts
                "MEDIUM": 0.0,
                "LOW": -0.02,     # 2% OTM for puts
            }

        offset = offsets.get(confidence_label, 0.0)
        raw_strike = current_price * (1 + offset)

        # Round to nearest standard strike increment
        if raw_strike < 20:
            increment = 0.50
        elif raw_strike < 100:
            increment = 1.0
        elif raw_strike < 500:
            increment = 5.0
        else:
            increment = 10.0

        return round(raw_strike / increment) * increment

    # ── Expiry Selection ─────────────────────────────────────────────────

    def select_expiry(
        self,
        hold_days: int,
        today: Optional[date] = None,
    ) -> tuple[str, int]:
        """
        Select expiry date: signal hold period + safety buffer.

        Buffer = max(hold_days * 0.5, 7 days) to avoid theta crush.
        Prefers standard monthly expirations (3rd Friday) for liquidity.
        Falls back to weekly Fridays only if no monthly is close enough.

        Returns (expiry_str, dte).
        """
        import calendar as _cal

        today = today or date.today()
        buffer = max(int(hold_days * 0.5), 7)
        target_dte = hold_days + buffer

        # Clamp to risk rules
        target_dte = max(target_dte, self.rules.min_dte_at_entry)
        target_dte = min(target_dte, self.rules.max_dte)

        # Collect 3rd-Friday monthlies for next 5 months
        monthlies: list[tuple[date, int]] = []
        for month_offset in range(0, 6):
            y = today.year + (today.month + month_offset - 1) // 12
            m = (today.month + month_offset - 1) % 12 + 1
            cal = _cal.monthcalendar(y, m)
            fridays = [week[_cal.FRIDAY] for week in cal if week[_cal.FRIDAY] != 0]
            tf = date(y, m, fridays[2])
            dte = (tf - today).days
            if dte >= 7:
                monthlies.append((tf, dte))

        # Pick the monthly closest to target_dte (>= 21 DTE preferred)
        viable = [(tf, dte) for tf, dte in monthlies if dte >= 21]
        if viable:
            viable.sort(key=lambda x: abs(x[1] - target_dte))
            expiry_date = viable[0][0]
            actual_dte = viable[0][1]
            return expiry_date.isoformat(), actual_dte

        # Any monthly >= 7 DTE
        if monthlies:
            monthlies.sort(key=lambda x: abs(x[1] - target_dte))
            expiry_date = monthlies[0][0]
            actual_dte = monthlies[0][1]
            return expiry_date.isoformat(), actual_dte

        # Last resort: nearest Friday >= 21 DTE
        target_date = today + timedelta(days=max(target_dte, 21))
        days_to_friday = (4 - target_date.weekday()) % 7
        if days_to_friday == 0 and target_date.weekday() != 4:
            days_to_friday = 7
        expiry_date = target_date + timedelta(days=days_to_friday)
        actual_dte = (expiry_date - today).days

        return expiry_date.isoformat(), actual_dte

    # ── Option Return Estimation ─────────────────────────────────────────

    @staticmethod
    def estimate_option_return(
        stock_move_pct: float,
        strike_offset_pct: float,
        dte: int,
        iv_estimate: float = 0.35,
    ) -> float:
        """
        Rough estimate of option return given stock move.

        Uses simplified delta-based model:
          - ATM call delta ~ 0.50
          - ITM call delta ~ 0.60-0.70
          - OTM call delta ~ 0.30-0.40
          - Option leverage = (stock_price * delta) / option_price

        For a small-account framework, this is a planning estimate,
        not a pricing model. Actual greeks come from Robinhood at execution.
        """
        # Approximate delta based on moneyness
        if strike_offset_pct <= -0.02:   # ITM
            delta = 0.65
        elif strike_offset_pct <= 0.005:  # ATM
            delta = 0.50
        elif strike_offset_pct <= 0.03:   # slightly OTM
            delta = 0.35
        else:                             # deep OTM
            delta = 0.20

        # Approximate option price as % of stock price
        # Using rough BS intuition: ATM option ~ stock_price * iv * sqrt(T/252) * 0.4
        t_years = dte / 365.0
        option_pct = iv_estimate * math.sqrt(t_years) * 0.4

        # Adjust for moneyness
        if strike_offset_pct < 0:  # ITM
            option_pct += abs(strike_offset_pct) * 0.8  # intrinsic value
        elif strike_offset_pct > 0:  # OTM
            option_pct *= max(0.3, 1 - strike_offset_pct * 10)  # cheaper

        option_pct = max(option_pct, 0.005)  # floor

        # Option return = delta * stock_move / option_cost_pct
        # Subtract some theta decay (rough: lose ~3% of remaining time value per day)
        leverage = delta / option_pct
        gross_return = stock_move_pct * leverage

        # Theta drag estimate (simplified)
        # Theta decay accelerates near expiry. For a hold of ~50% of DTE,
        # expect to lose ~8-12% of time value. Use 10% as middle estimate.
        theta_drag = 0.10

        net_return = gross_return - theta_drag
        return round(net_return * 100, 1)  # as percentage

    # ── Position Sizing ──────────────────────────────────────────────────

    def size_position(
        self,
        confidence_label: str,
        kelly_frac: float,
        estimated_option_price: float,
    ) -> tuple[int, float]:
        """
        Determine number of contracts and max cost.

        Returns (contracts, max_cost).
        """
        # Max allocation based on confidence
        # For small accounts (<$2000), use more aggressive caps so at least
        # 1 contract of reasonably priced options is tradeable
        if self.account_value < 2000:
            confidence_caps = {
                "HIGH": self.rules.max_single_trade_pct,   # 50% — need room for 1 contract
                "MEDIUM": self.rules.max_position_pct,     # 25%
                "LOW": self.rules.max_position_pct * 0.5,  # 12.5%
            }
        else:
            confidence_caps = {
                "HIGH": self.rules.max_position_pct,       # 25% of account
                "MEDIUM": self.rules.max_position_pct * 0.6, # 15% of account
                "LOW": self.rules.max_position_pct * 0.3,    # 7.5% of account
            }
        cap_pct = confidence_caps.get(confidence_label, 0.075)

        # Kelly-adjusted allocation
        kelly_alloc = self.account_value * kelly_frac
        cap_alloc = self.account_value * cap_pct

        # For small accounts, Kelly may be tiny — use cap as primary guide
        # and let Kelly act as a secondary filter only for larger accounts
        if self.account_value < 2000:
            max_alloc = cap_alloc
        else:
            max_alloc = min(kelly_alloc, cap_alloc)

        # Hard cap: no single trade > 50% of account
        max_alloc = min(max_alloc, self.account_value * self.rules.max_single_trade_pct)

        # Calculate contracts (each contract = 100 shares * option price)
        cost_per_contract = estimated_option_price * 100
        if cost_per_contract <= 0:
            return 0, 0.0

        contracts = max(1, int(max_alloc / cost_per_contract))

        # Verify total cost within limits
        total_cost = contracts * cost_per_contract
        while total_cost > max_alloc and contracts > 1:
            contracts -= 1
            total_cost = contracts * cost_per_contract

        # If even 1 contract exceeds 50% of account, skip
        # But for small accounts (<$2000), relax to allow up to max_single_trade_pct
        effective_cap = self.rules.max_single_trade_pct
        if cost_per_contract > self.account_value * effective_cap:
            return 0, 0.0

        return contracts, total_cost

    # ── Main Evaluation ──────────────────────────────────────────────────

    def evaluate_signals(
        self,
        ticker: str,
        current_price: float,
        signals: list[Signal],
        macro: Optional[MacroContext] = None,
        today: Optional[date] = None,
        iv_estimate: float = 0.35,
    ) -> TradeRecommendation:
        """
        Main entry point: evaluate signals and produce a trade recommendation.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        current_price : float
            Current stock price.
        signals : list[Signal]
            Active contrarian signals firing on this ticker.
        macro : MacroContext, optional
            Macro/sector context for confluence scoring.
        today : date, optional
            Override today's date (for testing).
        iv_estimate : float
            Estimated implied volatility (default 35%).

        Returns
        -------
        TradeRecommendation
            Complete trade decision with sizing, stops, and reasoning.
        """
        today = today or date.today()

        # Check position capacity
        if self.current_positions >= self.rules.max_concurrent_positions:
            return TradeRecommendation(
                action="SKIP",
                ticker=ticker,
                reasoning=f"At max concurrent positions ({self.rules.max_concurrent_positions})",
                signals_firing=[s.signal_type.value for s in signals],
            )

        # Compute confluence
        confluence_score, confidence = self.compute_confluence(signals, macro)

        # Gate: skip if confluence too low
        if confluence_score < 0.35:
            return TradeRecommendation(
                action="SKIP",
                ticker=ticker,
                confluence_score=confluence_score,
                confidence=confidence,
                signals_firing=[s.signal_type.value for s in signals],
                reasoning=f"Confluence too low ({confluence_score:.2f} < 0.35 threshold)",
            )

        # Determine direction (all contrarian signals are long by default)
        directions = [s.profile["direction"] for s in signals]
        direction = max(set(directions), key=directions.count)
        action = "BUY_CALL" if direction == "long" else "BUY_PUT"

        # Blended signal characteristics (weighted by Sharpe)
        total_w = sum(s.profile["sharpe"] for s in signals)
        blend_wr = sum(s.profile["base_win_rate"] * s.profile["sharpe"] for s in signals) / total_w
        blend_ret = sum(s.profile["expected_return_pct"] * s.profile["sharpe"] for s in signals) / total_w
        blend_hold = int(sum(s.profile["hold_days"] * s.profile["sharpe"] for s in signals) / total_w)

        # Strike selection
        strike = self.select_strike(current_price, blend_ret / 100, confidence, direction)
        strike_offset_pct = (strike - current_price) / current_price

        # Expiry selection
        expiry_str, dte = self.select_expiry(blend_hold, today)

        # Expected option return
        expected_opt_ret = self.estimate_option_return(
            blend_ret / 100, strike_offset_pct, dte, iv_estimate
        )

        # Kelly sizing
        # For options: avg win = expected option return, avg loss = stop loss (50%)
        avg_win = max(expected_opt_ret / 100, 0.10)  # floor at 10%
        avg_loss = self.rules.stop_loss_pct  # 50%
        kf = self.kelly_fraction(blend_wr, avg_win, avg_loss)

        # Estimate option price for sizing
        t_years = dte / 365.0
        est_opt_price_pct = iv_estimate * math.sqrt(t_years) * 0.4
        if strike_offset_pct < 0:
            est_opt_price_pct += abs(strike_offset_pct) * 0.8
        elif strike_offset_pct > 0:
            est_opt_price_pct *= max(0.3, 1 - strike_offset_pct * 10)
        est_opt_price = max(current_price * est_opt_price_pct, 0.10)

        # Size position
        contracts, max_cost = self.size_position(confidence, kf, est_opt_price)

        if contracts == 0:
            return TradeRecommendation(
                action="SKIP",
                ticker=ticker,
                confluence_score=confluence_score,
                confidence=confidence,
                signals_firing=[s.signal_type.value for s in signals],
                reasoning=f"Position too expensive for account size (est ${est_opt_price:.2f}/share, acct ${self.account_value:.0f})",
            )

        # Stop/target prices (in option premium terms)
        cost_per_contract = est_opt_price * 100
        stop_loss_price = cost_per_contract * (1 - self.rules.stop_loss_pct)
        take_profit_price = cost_per_contract * (1 + self.rules.take_profit_pct)

        risk_pct = (max_cost / self.account_value) * 100

        # Build reasoning
        signal_names = [s.signal_type.value for s in signals]
        indicator_summary = []
        for s in signals:
            for k, v in s.indicators.items():
                indicator_summary.append(f"{k}={v}")

        reasoning_parts = [
            f"{len(signals)} signal(s) firing: {', '.join(signal_names)}.",
            f"Blended WR={blend_wr:.0%}, expected stock move=+{blend_ret:.1f}%.",
        ]
        if indicator_summary:
            reasoning_parts.append(f"Indicators: {', '.join(indicator_summary)}.")
        if macro:
            macro_notes = []
            if macro.sector_momentum_aligned:
                macro_notes.append("sector aligned")
            if macro.breadth_positive:
                macro_notes.append("breadth positive")
            if macro.cta_flow_supportive:
                macro_notes.append("CTA supportive")
            if macro_notes:
                reasoning_parts.append(f"Macro: {', '.join(macro_notes)}.")
            if macro.vix_regime != "normal":
                reasoning_parts.append(f"VIX regime: {macro.vix_regime}.")

        return TradeRecommendation(
            action=action,
            ticker=ticker,
            strike=strike,
            expiry=expiry_str,
            contracts=contracts,
            max_cost=round(max_cost, 2),
            confidence=confidence,
            confluence_score=round(confluence_score, 3),
            signals_firing=signal_names,
            expected_stock_move_pct=round(blend_ret, 1),
            expected_option_return_pct=round(expected_opt_ret, 1),
            stop_loss_price=round(stop_loss_price, 2),
            take_profit_price=round(take_profit_price, 2),
            time_stop_dte=self.rules.time_stop_dte,
            reasoning=" ".join(reasoning_parts),
            risk_pct_of_account=round(risk_pct, 1),
            kelly_fraction=round(kf, 4),
        )

    # ── Batch Evaluation ─────────────────────────────────────────────────

    def evaluate_batch(
        self,
        candidates: list[dict],
        macro: Optional[MacroContext] = None,
        today: Optional[date] = None,
    ) -> list[TradeRecommendation]:
        """
        Evaluate multiple candidates, respecting position limits.

        Each candidate dict:
            {"ticker": "AAPL", "price": 195.0, "signals": [...], "iv": 0.30}

        Returns sorted by confluence score (highest first), filtered by risk rules.
        """
        recs = []
        for cand in candidates:
            rec = self.evaluate_signals(
                ticker=cand["ticker"],
                current_price=cand["price"],
                signals=cand["signals"],
                macro=macro,
                today=today,
                iv_estimate=cand.get("iv", 0.35),
            )
            recs.append(rec)

        # Sort by confluence (best first)
        recs.sort(key=lambda r: r.confluence_score, reverse=True)

        # Apply position limit: only take top N non-SKIP
        available_slots = self.rules.max_concurrent_positions - self.current_positions
        taken = 0
        final = []
        for rec in recs:
            if rec.action != "SKIP" and taken >= available_slots:
                rec = TradeRecommendation(
                    action="SKIP",
                    ticker=rec.ticker,
                    confluence_score=rec.confluence_score,
                    confidence=rec.confidence,
                    signals_firing=rec.signals_firing,
                    reasoning=f"Position limit reached ({self.rules.max_concurrent_positions} max)",
                )
            if rec.action != "SKIP":
                taken += 1
            final.append(rec)

        return final


# ── Example Scenarios ────────────────────────────────────────────────────────

def run_example_scenarios() -> list[dict]:
    """
    Run 5 example scenarios demonstrating the framework.
    Returns list of scenario dicts for JSON export.
    """
    fw = OptionsFramework(account_value=677.0, current_positions=0)
    scenarios = []
    test_date = date(2026, 7, 23)

    # ── Scenario 1: Single signal - momentum exhaustion on INTC (cheap stock) ──
    rec1 = fw.evaluate_signals(
        ticker="INTC",
        current_price=24.80,
        signals=[
            Signal(
                signal_type=SignalType.MOMENTUM_EXHAUSTION,
                confidence=0.58,
                indicators={"RSI_14": 28, "MFI_14": 18, "price_drop_5d": "-4.2%"},
            ),
        ],
        macro=MacroContext(
            sector_momentum_aligned=True,
            vix_regime="normal",
            risk_appetite="neutral",
        ),
        today=test_date,
        iv_estimate=0.45,
    )
    scenarios.append({
        "name": "Single signal: Momentum exhaustion on INTC",
        "description": "INTC RSI=28, MFI=18 after 4.2% 5-day drop. Single oversold signal. Cheap stock suitable for small account.",
        "recommendation": rec1.to_dict(),
        "summary": rec1.summary(),
    })

    # ── Scenario 2: Double confluence on PFE post-earnings (affordable) ──
    rec2 = fw.evaluate_signals(
        ticker="PFE",
        current_price=26.50,
        signals=[
            Signal(
                signal_type=SignalType.POST_EARNINGS_DRIFT,
                confidence=0.72,
                indicators={"earnings_drop": "-12%", "days_since_event": 22, "volume_mult": 2.8},
            ),
            Signal(
                signal_type=SignalType.SMART_MONEY,
                confidence=0.68,
                indicators={"OBV_trend": "rising", "MFI_14": 16, "insider_buys": 3},
            ),
        ],
        macro=MacroContext(
            sector_momentum_aligned=True,
            cta_flow_supportive=True,
            breadth_positive=True,
            vix_regime="normal",
            risk_appetite="risk_on",
        ),
        today=test_date,
        iv_estimate=0.32,
    )
    scenarios.append({
        "name": "Double confluence: Post-earnings drift + smart money on PFE",
        "description": "PFE dropped 12% on earnings, now 22 days later with OBV rising + insider buying. Strong macro support. Affordable for small account.",
        "recommendation": rec2.to_dict(),
        "summary": rec2.summary(),
    })

    # ── Scenario 3: Sector ETF play on XLE ───────────────────────────
    rec3 = fw.evaluate_signals(
        ticker="XLE",
        current_price=91.50,
        signals=[
            Signal(
                signal_type=SignalType.SECTOR_MOMENTUM,
                confidence=0.60,
                indicators={"sector_rank": 2, "rel_strength_20d": 1.08, "flow_score": 0.72},
            ),
            Signal(
                signal_type=SignalType.PRICE_VOLUME_DIVERGENCE,
                confidence=0.55,
                indicators={"OBV_slope": "up", "MFI_14": 32, "RSI_14": 38},
            ),
        ],
        macro=MacroContext(
            sector_momentum_aligned=True,
            breadth_positive=True,
            vix_regime="low",
            risk_appetite="risk_on",
        ),
        today=test_date,
        iv_estimate=0.25,
    )
    scenarios.append({
        "name": "Sector play: XLE strong momentum + price-volume divergence",
        "description": "Energy sector ranked #2 with strong relative strength. XLE showing OBV rising while price pulled back. Low VIX, risk-on.",
        "recommendation": rec3.to_dict(),
        "summary": rec3.summary(),
    })

    # ── Scenario 4: Triple confluence on CMCSA (affordable) ─────────
    rec4 = fw.evaluate_signals(
        ticker="CMCSA",
        current_price=21.90,
        signals=[
            Signal(
                signal_type=SignalType.SKEWNESS_PREMIUM,
                confidence=0.62,
                indicators={"skew_1y": -1.17, "drop_1d": "-6.8%"},
            ),
            Signal(
                signal_type=SignalType.PRICE_VOLUME_DIVERGENCE,
                confidence=0.59,
                indicators={"OBV_slope": "up", "MFI_14": 22, "RSI_14": 31},
            ),
            Signal(
                signal_type=SignalType.VOL_CRUSH_REVERSAL,
                confidence=0.53,
                indicators={"ATR_pct": 15, "drop_5d": "-8.1%", "MFI_14": 19},
            ),
        ],
        macro=MacroContext(
            sector_momentum_aligned=True,
            cta_flow_supportive=True,
            vix_regime="normal",
            risk_appetite="neutral",
        ),
        today=test_date,
        iv_estimate=0.38,
    )
    scenarios.append({
        "name": "Triple confluence: Skewness + price-volume div + vol crush on CMCSA",
        "description": "CMCSA dropped 6.8% on negative skew, OBV rising, ATR compressed. 3 independent signals agreeing. Currently in paper portfolio.",
        "recommendation": rec4.to_dict(),
        "summary": rec4.summary(),
    })

    # ── Scenario 5: SKIP - weak signal, bad macro ────────────────────
    rec5 = fw.evaluate_signals(
        ticker="NKE",
        current_price=68.50,
        signals=[
            Signal(
                signal_type=SignalType.VOL_CRUSH_REVERSAL,
                confidence=0.35,
                indicators={"ATR_pct": 30, "drop_5d": "-3.1%", "MFI_14": 28},
            ),
        ],
        macro=MacroContext(
            sector_momentum_aligned=False,
            cta_flow_supportive=False,
            breadth_positive=False,
            vix_regime="elevated",
            risk_appetite="risk_off",
        ),
        today=test_date,
        iv_estimate=0.40,
    )
    scenarios.append({
        "name": "SKIP: Weak vol crush signal on NKE, bad macro",
        "description": "NKE has a single weak vol crush signal (confidence 0.35). VIX elevated, risk-off, no sector support. Framework correctly skips.",
        "recommendation": rec5.to_dict(),
        "summary": rec5.summary(),
    })

    return scenarios


# ── Integration Notes ────────────────────────────────────────────────────────

INTEGRATION_PLAN = """
INTEGRATION PLAN
================

1. CONTRARIAN SIGNALS PAPER ENGINE (fires at 3:50 PM ET daily)
   - Engine: /home/jupiter/Lvl3Quant/engines/contrarian_signals_paper.py
   - State: /home/jupiter/Lvl3Quant/state/contrarian_signals_state.json
   - When signals fire, construct Signal objects and call fw.evaluate_signals()
   - The paper engine detects: skewness, post-earnings drift, smart money,
     price-volume divergence, vol crush reversal

2. EXPANDED ALERTER (15 active signals)
   - State: /home/jupiter/Lvl3Quant/state/expanded_alerter_state.json
   - Additional signals beyond the 5 core contrarian signals
   - Feed into this framework as supplementary confluence

3. SECTOR ANALYSIS MODULE
   - Provides MacroContext.sector_momentum_aligned
   - Sector rank, relative strength, flow scores feed into confluence

4. ROBINHOOD MCP TOOLS (execution layer)
   - get_option_chains -> actual strikes/expiries/prices
   - get_option_quotes -> real IV, greeks, bid-ask
   - review_option_order -> pre-flight check
   - place_option_order -> execute the trade
   - get_option_positions -> monitor open positions

5. WORKFLOW:
   a. 3:50 PM: Contrarian engine fires, detects signals across universe
   b. Framework evaluates each signal (or signal cluster per ticker)
   c. If recommendation != SKIP:
      i.   Fetch real option chain from Robinhood
      ii.  Validate strike exists, check bid-ask spread
      iii. If spread > 10% of premium, widen strike or skip
      iv.  Review order via review_option_order
      v.   Place order if review passes
   d. Position management:
      i.   Daily check: stop loss (50% premium loss), take profit (100% gain)
      ii.  Time stop: exit within 3 DTE
      iii. Update state file with positions
"""


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scenarios = run_example_scenarios()

    print("=" * 80)
    print("AGENTIC OPTIONS FRAMEWORK v1 - Example Scenarios")
    print(f"Account: $677 | Max positions: 3 | Date: 2026-07-23")
    print("=" * 80)

    for i, sc in enumerate(scenarios, 1):
        print(f"\n{'─' * 70}")
        print(f"SCENARIO {i}: {sc['name']}")
        print(f"{'─' * 70}")
        print(f"Context: {sc['description']}")
        print(f"\nResult: {sc['summary']}")

        rec = sc["recommendation"]
        if rec["action"] != "SKIP":
            print(f"\n  Action:          {rec['action']}")
            print(f"  Ticker:          {rec['ticker']}")
            print(f"  Strike:          ${rec['strike']}")
            print(f"  Expiry:          {rec['expiry']} ({rec.get('time_stop_dte', 3)} DTE time-stop)")
            print(f"  Contracts:       {rec['contracts']}")
            print(f"  Max Cost:        ${rec['max_cost']:.2f}")
            print(f"  Confluence:      {rec['confluence_score']:.3f} ({rec['confidence']})")
            print(f"  Kelly Fraction:  {rec['kelly_fraction']:.4f}")
            print(f"  Exp Stock Move:  +{rec['expected_stock_move_pct']}%")
            print(f"  Exp Option Ret:  +{rec['expected_option_return_pct']}%")
            print(f"  Stop Loss:       ${rec['stop_loss_price']:.2f} per contract")
            print(f"  Take Profit:     ${rec['take_profit_price']:.2f} per contract")
            print(f"  Risk % of Acct:  {rec['risk_pct_of_account']}%")
            print(f"  Signals:         {', '.join(rec['signals_firing'])}")
        else:
            print(f"\n  Action:          SKIP")
            print(f"  Confluence:      {rec['confluence_score']:.3f}")
            print(f"  Reason:          {rec['reasoning']}")

    print(f"\n{'=' * 80}")
    print(INTEGRATION_PLAN)

    # Save examples to JSON
    output_path = "/home/jupiter/Lvl3Quant/research/findings/agentic_framework_examples.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "framework": "agentic_options_framework_v1",
                "account_value": 677.0,
                "date": "2026-07-23",
                "scenarios": scenarios,
                "integration_plan": INTEGRATION_PLAN.strip(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nExamples saved to {output_path}")
