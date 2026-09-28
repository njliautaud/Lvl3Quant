"""
Setup Analyzer - Find setups that work CONSISTENTLY
====================================================

Model predictions are useless without good execution.
This finds the specific market conditions where we:
1. Get good fills (tight spread, good queue position)
2. Win consistently (not 60% overall, but 80%+ in specific setups)
3. Have favorable risk/reward (wins > losses)

Focus: QUALITY over QUANTITY. Trade 10 great setups/day, not 1000 mediocre ones.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict

@dataclass
class Setup:
    """A specific market condition that leads to consistent wins"""
    name: str
    conditions: dict
    win_rate: float
    avg_win_ticks: float
    avg_loss_ticks: float
    profit_factor: float
    fill_rate: float
    sample_count: int
    sharpe: float

class SetupAnalyzer:
    """
    Analyze predictions to find the BEST setups, not just average performance.

    Key insight: 60% direction accuracy overall is useless if:
    - 90% accuracy in tight spread, high volume conditions (but rare)
    - 40% accuracy in wide spread, low volume (but common)

    We want to ONLY trade the 90% setups, ignore the 40% ones.
    """

    def __init__(self):
        self.setups = []

    def define_setups(self) -> List[dict]:
        """
        Define candidate setups to test.
        Each setup = specific market conditions.
        """
        return [
            # Tight spread, high confidence
            {
                'name': 'Tight_Spread_High_Conf',
                'spread_max': 1.0,
                'confidence_min': 0.8,
                'volume_min': None
            },
            # Large OFI imbalance, tight spread
            {
                'name': 'Large_OFI_Tight',
                'ofi_min': 100,
                'spread_max': 1.5,
                'confidence_min': 0.6
            },
            # Post-trade aggression (continuation)
            {
                'name': 'Trade_Continuation',
                'recent_trades_same_side': 3,
                'spread_max': 2.0,
                'confidence_min': 0.5
            },
            # Queue depth advantage (top of book)
            {
                'name': 'Queue_Advantage',
                'our_queue_position': 'top_3',
                'spread_max': 1.5,
                'confidence_min': 0.7
            },
            # Reversal setups (mean reversion)
            {
                'name': 'Reversal_Extreme',
                'recent_move_ticks': 5,
                'ofi_reversal': True,
                'spread_max': 2.0,
                'confidence_min': 0.6
            },
        ]

    def analyze_setup(self,
                     predictions: np.ndarray,
                     labels: np.ndarray,
                     market_conditions: dict) -> Setup:
        """
        Analyze how well a specific setup performs.

        Returns metrics for this setup:
        - Win rate
        - Avg win/loss
        - Fill probability
        - Risk-adjusted returns
        """

        # Filter predictions to this setup's conditions
        mask = self._apply_conditions(market_conditions)

        if mask.sum() < 50:  # Need enough samples
            return None

        setup_preds = predictions[mask]
        setup_labels = labels[mask]

        # Win rate
        correct = np.sign(setup_preds) == np.sign(setup_labels)
        win_rate = correct.mean()

        # P&L distribution
        wins = setup_labels[correct]
        losses = setup_labels[~correct]

        avg_win = abs(wins.mean()) * 1e4 if len(wins) > 0 else 0  # to ticks
        avg_loss = abs(losses.mean()) * 1e4 if len(losses) > 0 else 0

        # Profit factor
        total_wins = abs(wins.sum()) * 1e4
        total_losses = abs(losses.sum()) * 1e4
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Fill rate (simplified - assume better for tighter spreads)
        avg_spread = market_conditions.get('avg_spread', 2.0)
        fill_rate = max(0.5, min(1.0, 2.0 / avg_spread))  # Higher for tight spreads

        # Sharpe (simplified)
        returns = np.where(correct, avg_win, -avg_loss)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        return Setup(
            name=market_conditions.get('name', 'Unknown'),
            conditions=market_conditions,
            win_rate=win_rate,
            avg_win_ticks=avg_win,
            avg_loss_ticks=avg_loss,
            profit_factor=profit_factor,
            fill_rate=fill_rate,
            sample_count=mask.sum(),
            sharpe=sharpe
        )

    def find_best_setups(self,
                        predictions: np.ndarray,
                        labels: np.ndarray,
                        market_data: dict) -> List[Setup]:
        """
        Find the top setups we should actually trade.

        Criteria:
        - Win rate > 65%
        - Profit factor > 2.0
        - Fill rate > 70%
        - Sharpe > 2.0
        - Sample count > 100
        """

        candidate_setups = self.define_setups()
        validated_setups = []

        for setup_def in candidate_setups:
            setup = self.analyze_setup(predictions, labels, setup_def)

            if setup and self._meets_criteria(setup):
                validated_setups.append(setup)

        # Sort by Sharpe (risk-adjusted)
        validated_setups.sort(key=lambda s: s.sharpe, reverse=True)

        return validated_setups

    def _meets_criteria(self, setup: Setup) -> bool:
        """Check if setup meets deployment criteria"""
        return (
            setup.win_rate >= 0.65 and
            setup.profit_factor >= 2.0 and
            setup.fill_rate >= 0.70 and
            setup.sharpe >= 2.0 and
            setup.sample_count >= 100
        )

    def _apply_conditions(self, conditions: dict) -> np.ndarray:
        """Apply setup conditions to create boolean mask"""
        # Simplified - would need actual market data
        # For now, return random mask as placeholder
        return np.random.random(1000) > 0.5

    def print_setup_report(self, setups: List[Setup]):
        """Print deployable setups"""
        print(f"\n{'='*70}")
        print("DEPLOYABLE SETUPS (Quality over Quantity)")
        print(f"{'='*70}\n")

        if not setups:
            print("❌ No setups meet criteria.")
            print("Need: Win rate ≥65%, Profit factor ≥2.0, Fill rate ≥70%, Sharpe ≥2.0")
            return

        for i, setup in enumerate(setups, 1):
            print(f"{i}. {setup.name}")
            print(f"   Win Rate:      {setup.win_rate:.1%}")
            print(f"   Profit Factor: {setup.profit_factor:.2f}")
            print(f"   Fill Rate:     {setup.fill_rate:.1%}")
            print(f"   Sharpe:        {setup.sharpe:.2f}")
            print(f"   Avg Win/Loss:  {setup.avg_win_ticks:.1f} / {setup.avg_loss_ticks:.1f} ticks")
            print(f"   Samples:       {setup.sample_count:,}")
            print()

        print(f"Trade ONLY these {len(setups)} setups. Ignore everything else.")
        print("Quality > Quantity.")


if __name__ == "__main__":
    print("Setup Analyzer - Execution Optimization")
    print("\nPhilosophy:")
    print("  Models are 50% of the battle")
    print("  Execution optimization is the other 50%")
    print("  Trade 10 great setups/day, not 1000 mediocre ones")
    print("\nFocus:")
    print("  1. Find setups with 80%+ win rate (not 60% overall)")
    print("  2. Ensure good fills (tight spread, good queue position)")
    print("  3. Consistency over volume")
