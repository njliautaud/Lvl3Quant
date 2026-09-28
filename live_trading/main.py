"""
Paper Trading Engine — Main Orchestrator
=========================================

Wires together all components:
1. Data feed (replay or live)
2. Inference engine (models)
3. Card engine (strategies)
4. Fill simulator
5. Position manager
6. Monitoring & alerts

Usage:
    # Replay mode (for testing)
    python -m live_trading.main --mode replay --config configs/paper_config.json

    # Live mode (production)
    python -m live_trading.main --mode live --config configs/live_config.json
"""

import asyncio
import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Local imports
from live_trading.data_feed import create_feed, MBOEvent
from live_trading.model_registry import get_registry
from live_trading.inference_engine import InferenceEngine, Prediction
from live_trading.card_engine import CardEngine, CardConfig
from live_trading.fill_simulator import FillSimulator, Fill
from live_trading.position_manager import PositionManager
from live_trading.safety import SafetyManager

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Main orchestrator for paper trading"""

    def __init__(self, config: dict):
        self.config = config
        self._running = False
        self._stop_event = asyncio.Event()

        # Initialize components
        self.feed = None
        self.inference = InferenceEngine(
            event_window_size=config.get('event_window_size', 500),
            warmup_events=config.get('warmup_events', 500),
            prediction_stride=config.get('prediction_stride', 1),
        )
        self.cards = CardEngine()
        self.fill_sim = FillSimulator(
            market_slippage_ticks=config.get('market_slippage_ticks', 0.5),
            limit_fill_probability=config.get('limit_fill_probability', 0.7),
            latency_ms=config.get('latency_ms', 10.0),
        )
        self.positions = PositionManager(
            max_total_position=config.get('max_total_position', 5),
            max_card_position=config.get('max_card_position', 1),
        )

        # Safety layer
        self.safety = SafetyManager(config)

        # Metrics
        self._events_processed = 0
        self._predictions_generated = 0
        self._predictions_rejected = 0
        self._orders_submitted = 0
        self._orders_rejected = 0
        self._fills_received = 0
        self._last_status_time = 0.0

        # Setup callbacks
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Wire up callbacks between components"""
        # Inference -> Cards
        def on_prediction(prediction: Prediction):
            self._predictions_generated += 1
            current_price = self.inference.get_last_mid()
            self.cards.on_prediction(prediction, current_price)

        # Register callback for each model (will be set up after models loaded)
        # Done dynamically in start()

        # Cards -> Fill Simulator
        def on_order(card_name: str, side: str, size: int, price_type: str, limit_price: Optional[float]):
            self._orders_submitted += 1
            self.fill_sim.submit_order(
                card_name=card_name,
                side=side,
                size=size,
                order_type=price_type,
                limit_price=limit_price,
            )

        # Fill Simulator -> Positions & Cards
        def on_fill(fill: Fill):
            self._fills_received += 1
            card = self.cards.get_card(fill.card_name)
            if card:
                # Check if opening or closing position
                if self.positions.has_position(fill.card_name):
                    # Closing position
                    self.positions.close_position(fill.card_name, fill.price, "fill")
                else:
                    # Opening position
                    side = "LONG" if fill.side == "BUY" else "SHORT"
                    self.positions.open_position(fill.card_name, side, fill.size, fill.price)
                    card.on_fill(fill.price, fill.size)

        self.fill_sim.set_fill_callback(on_fill)

        # Set order callback for all cards
        for card in self.cards.cards.values():
            card.set_order_callback(on_order)

    async def load_config(self):
        """Load models and cards from config"""
        registry = get_registry()

        # Load models
        logger.info("Loading models...")
        for model_cfg in self.config.get('models', []):
            try:
                registry.register_model(
                    name=model_cfg['name'],
                    model_path=model_cfg['path'],
                    model_type=model_cfg['type'],
                    architecture=model_cfg['architecture'],
                    fold=model_cfg.get('fold', 0),
                    label_horizon=model_cfg.get('label_horizon', '10s'),
                    ic_score=model_cfg.get('ic_score', 0.0),
                )

                # Register prediction callback with safety validation
                def make_callback(model_name):
                    def callback(pred: Prediction):
                        self._predictions_generated += 1

                        # SAFETY: Validate prediction before using
                        valid, reason = self.safety.validate_prediction(pred)
                        if not valid:
                            self._predictions_rejected += 1
                            logger.warning(f"Prediction rejected: {reason}")
                            return

                        current_price = self.inference.get_last_mid()
                        self.cards.on_prediction(pred, current_price)
                    return callback

                self.inference.on_prediction(model_cfg['name'], make_callback(model_cfg['name']))

                logger.info(f"  ✓ {model_cfg['name']} ({model_cfg['architecture']})")
            except Exception as e:
                logger.error(f"  ✗ Failed to load {model_cfg['name']}: {e}")

        # Load cards
        logger.info("Loading trading cards...")
        for card_cfg in self.config.get('cards', []):
            try:
                config = CardConfig(
                    name=card_cfg['name'],
                    model_name=card_cfg['model_name'],
                    threshold=card_cfg.get('threshold', 0.0),
                    min_tier=card_cfg.get('min_tier', 'all'),
                    max_position_size=card_cfg.get('max_position_size', 1),
                    take_profit_ticks=card_cfg.get('take_profit_ticks'),
                    stop_loss_ticks=card_cfg.get('stop_loss_ticks'),
                    max_hold_seconds=card_cfg.get('max_hold_seconds'),
                    conviction_decay=card_cfg.get('conviction_decay', False),
                    time_filter=card_cfg.get('time_filter'),
                    vol_filter=card_cfg.get('vol_filter'),
                    chase_max_ticks=card_cfg.get('chase_max_ticks', 1),
                    chase_max_reprices=card_cfg.get('chase_max_reprices', 3),
                    enabled=card_cfg.get('enabled', True),
                    description=card_cfg.get('description', ''),
                )
                card = self.cards.add_card(config)
                card.set_order_callback(lambda cn, s, sz, pt, lp: self.fill_sim.submit_order(cn, s, sz, pt, lp))
                logger.info(f"  ✓ {card_cfg['name']} (model={card_cfg['model_name']})")
            except Exception as e:
                logger.error(f"  ✗ Failed to load card {card_cfg['name']}: {e}")

    async def start(self):
        """Start paper trading engine"""
        logger.info("=" * 60)
        logger.info("PAPER TRADING ENGINE STARTING")
        logger.info("=" * 60)

        # Load configuration
        await self.load_config()

        # Create data feed
        feed_config = self.config.get('feed', {})
        self.feed = create_feed(
            mode=feed_config.get('mode', 'replay'),
            data_dir=feed_config.get('data_dir'),
            replay_speed=feed_config.get('replay_speed', 1.0),
            start_date=feed_config.get('start_date'),
            end_date=feed_config.get('end_date'),
        )

        # Connect feed
        await self.feed.connect()
        await self.feed.subscribe(
            symbol=feed_config.get('symbol', 'ES'),
            exchange=feed_config.get('exchange', 'CME'),
        )

        # Start processing
        self._running = True
        logger.info("Engine ready. Streaming data...")
        logger.info("")

        try:
            await self._run_loop()
        finally:
            await self._shutdown()

    async def _run_loop(self):
        """Main event processing loop"""
        async for event in self.feed.stream():
            if not self._running:
                break

            # SAFETY: Check kill switch and circuit breakers
            should_halt, halt_reason = self.safety.should_halt()
            if should_halt:
                logger.critical(f"HALTING TRADING: {halt_reason}")
                await self._emergency_shutdown(halt_reason)
                break

            # SAFETY: Check positions for risk violations (every 100 events)
            if self._events_processed % 100 == 0:
                positions_to_close = self.safety.check_positions(self.positions.positions)
                for card_name, reason in positions_to_close:
                    logger.warning(f"Force-closing position {card_name}: {reason}")
                    # Force close would go here (not implemented in position_manager yet)

            # Update fill simulator with market state
            if event.event_type == 0:  # Add event
                if event.side == 0:  # Bid
                    self.fill_sim.update_market_state(
                        best_bid=event.price,
                        best_ask=self.fill_sim._best_ask or event.price + event.spread,
                    )
                elif event.side == 1:  # Ask
                    self.fill_sim.update_market_state(
                        best_bid=self.fill_sim._best_bid or event.price - event.spread,
                        best_ask=event.price,
                    )

            # Process event through inference engine
            predictions = await self.inference.process_event(event)

            # Update position marks
            current_mid = self.inference.get_last_mid()
            if current_mid > 0:
                self.positions.update_mark_price(current_mid)

            self._events_processed += 1

            # Periodic status logging
            if self._events_processed % 10000 == 0:
                self._log_status()

    def _log_status(self):
        """Log current status"""
        import time
        now = time.time()
        if now - self._last_status_time < 60:  # Don't spam logs
            return

        self._last_status_time = now

        pnl_summary = self.positions.get_summary()
        card_summary = self.cards.get_summary()

        logger.info("=" * 60)
        logger.info(f"STATUS @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Events: {self._events_processed:,} | Predictions: {self._predictions_generated:,} | Orders: {self._orders_submitted} | Fills: {self._fills_received}")
        logger.info(f"Positions: {pnl_summary['open_positions']} | Exposure: {pnl_summary['total_exposure']}")
        logger.info(f"Unrealized PnL: {pnl_summary['unrealized_pnl_ticks']:+.1f}t (${pnl_summary['unrealized_pnl_dollars']:+,.2f})")
        logger.info(f"Realized PnL: {pnl_summary['realized_pnl_ticks']:+.1f}t (${pnl_summary['realized_pnl_dollars']:+,.2f})")
        logger.info(f"Total PnL: {pnl_summary['total_pnl_ticks']:+.1f}t (${pnl_summary['total_pnl_dollars']:+,.2f})")
        logger.info(f"Trades: {pnl_summary['total_trades']} (W/L: {pnl_summary['winning_trades']}/{pnl_summary['losing_trades']}, WR: {pnl_summary['win_rate']:.1%})")
        logger.info("")
        for card_name, stats in card_summary.items():
            logger.info(f"  {card_name}: {stats['state']} | {stats['total_trades']} trades | PnL: {stats['total_pnl_ticks']:+.1f}t")
        logger.info("=" * 60)

    async def _shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down...")
        self._running = False

        # Disconnect feed
        if self.feed:
            await self.feed.disconnect()

        # Final status
        self._log_status()

        # Save final report
        self._save_final_report()

        logger.info("Shutdown complete.")

    def _save_final_report(self):
        """Save final PnL report"""
        report_path = Path("/home/jupiter/Lvl3Quant/live_trading/logs/final_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "metrics": {
                "events_processed": self._events_processed,
                "predictions_generated": self._predictions_generated,
                "orders_submitted": self._orders_submitted,
                "fills_received": self._fills_received,
            },
            "pnl": self.positions.get_summary(),
            "cards": self.cards.get_summary(),
            "recent_trades": self.positions.get_recent_trades(50),
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Final report saved: {report_path}")

    def stop(self):
        """Stop engine"""
        self._running = False
        self._stop_event.set()


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path("/home/jupiter/Lvl3Quant/live_trading/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"engine_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )

    logger.info(f"Logging to: {log_file}")


async def main_async(args):
    """Async main entry point"""
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # Override with CLI args
    if args.mode:
        config.setdefault('feed', {})['mode'] = args.mode

    # Create engine
    engine = PaperTradingEngine(config)

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, stopping...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start engine
    await engine.start()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Paper Trading Engine")
    parser.add_argument('--config', required=True, help="Path to config JSON file")
    parser.add_argument('--mode', choices=['replay', 'live'], help="Override feed mode")
    parser.add_argument('--log-level', default='INFO', help="Logging level")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Run
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
