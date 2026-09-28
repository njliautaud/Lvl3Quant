#!/usr/bin/env python3
"""
Quick System Test Script
========================

Validates all components work correctly without running full simulation.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all modules can be imported"""
    print("Testing imports...")
    try:
        from live_trading import data_feed
        from live_trading import model_registry
        from live_trading import inference_engine
        from live_trading import card_engine
        from live_trading import fill_simulator
        from live_trading import position_manager
        from live_trading import main
        print("  ✓ All modules import successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_data_feed():
    """Test data feed can load data"""
    print("\nTesting data feed...")
    try:
        from live_trading.data_feed import create_feed
        import asyncio

        async def test():
            feed = create_feed(
                mode="replay",
                data_dir="/home/jupiter/Lvl3Quant/data/processed/mbo_events",
                replay_speed=0.0,
                start_date="20260115",
                end_date="20260115",
            )
            await feed.connect()

            event_count = 0
            async for event in feed.stream():
                event_count += 1
                if event_count >= 1000:  # Test first 1000 events
                    break

            await feed.disconnect()
            return event_count

        count = asyncio.run(test())
        print(f"  ✓ Successfully streamed {count} events")
        return True
    except Exception as e:
        print(f"  ✗ Data feed test failed: {e}")
        return False


def test_model_registry():
    """Test model registry can register models"""
    print("\nTesting model registry...")
    try:
        from live_trading.model_registry import get_registry

        registry = get_registry()

        # Try to find a model file
        model_paths = list(Path("/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/event_cnn_1d").rglob("model_final.pt"))

        if not model_paths:
            print("  ⚠ No EventCNN1D model files found (this is OK if not trained yet)")
            return True

        # Register first model found
        model_path = str(model_paths[0])
        registry.register_model(
            name="test_model",
            model_path=model_path,
            model_type="pytorch",
            architecture="event_cnn_1d",
            fold=0,
            label_horizon="10s",
            ic_score=0.0,
        )

        model = registry.get_model("test_model")
        if model:
            print(f"  ✓ Model registered successfully: {model_path}")
            return True
        else:
            print("  ✗ Model registration failed")
            return False

    except Exception as e:
        print(f"  ✗ Model registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_engine():
    """Test inference engine"""
    print("\nTesting inference engine...")
    try:
        from live_trading.inference_engine import InferenceEngine
        from live_trading.data_feed import MBOEvent
        import asyncio

        engine = InferenceEngine()

        # Create dummy event
        event = MBOEvent(
            timestamp=1.0,
            event_type=0,
            side=0,
            price=5000.0,
            qty=10.0,
            spread=1.0,
            time_delta=0.001,
        )

        # Process event
        predictions = asyncio.run(engine.process_event(event))

        print(f"  ✓ Inference engine initialized (processed 1 event)")
        return True

    except Exception as e:
        print(f"  ✗ Inference engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_card_engine():
    """Test card engine"""
    print("\nTesting card engine...")
    try:
        from live_trading.card_engine import CardEngine, CardConfig

        cards = CardEngine()

        config = CardConfig(
            name="test_card",
            model_name="test_model",
            threshold=2.0,
            max_position_size=1,
        )

        card = cards.add_card(config)

        if card and cards.get_card("test_card"):
            print(f"  ✓ Card engine working (added 1 card)")
            return True
        else:
            print("  ✗ Card engine test failed")
            return False

    except Exception as e:
        print(f"  ✗ Card engine test failed: {e}")
        return False


def test_fill_simulator():
    """Test fill simulator"""
    print("\nTesting fill simulator...")
    try:
        from live_trading.fill_simulator import FillSimulator

        sim = FillSimulator()
        sim.update_market_state(best_bid=5000.0, best_ask=5000.25)

        order_id = sim.submit_order(
            card_name="test_card",
            side="BUY",
            size=1,
            order_type="MARKET",
        )

        if order_id:
            print(f"  ✓ Fill simulator working (order ID: {order_id})")
            return True
        else:
            print("  ✗ Fill simulator test failed")
            return False

    except Exception as e:
        print(f"  ✗ Fill simulator test failed: {e}")
        return False


def test_position_manager():
    """Test position manager"""
    print("\nTesting position manager...")
    try:
        from live_trading.position_manager import PositionManager

        pm = PositionManager()

        success = pm.open_position(
            card_name="test_card",
            side="LONG",
            size=1,
            entry_price=5000.0,
        )

        if success:
            pm.update_mark_price(5001.0)
            summary = pm.get_summary()
            print(f"  ✓ Position manager working (PnL: ${summary['unrealized_pnl_dollars']:.2f})")
            return True
        else:
            print("  ✗ Position manager test failed")
            return False

    except Exception as e:
        print(f"  ✗ Position manager test failed: {e}")
        return False


def test_config_files():
    """Test config files are valid JSON"""
    print("\nTesting config files...")
    try:
        import json

        config_dir = Path("/home/jupiter/Lvl3Quant/live_trading/configs")
        configs = list(config_dir.glob("*.json"))

        for config_path in configs:
            with open(config_path) as f:
                json.load(f)
            print(f"  ✓ {config_path.name} is valid")

        return True

    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("PAPER TRADING SYSTEM TEST")
    print("=" * 60)

    tests = [
        test_imports,
        test_config_files,
        test_data_feed,
        test_inference_engine,
        test_card_engine,
        test_fill_simulator,
        test_position_manager,
        # test_model_registry,  # Skip if models not available yet
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! System ready for deployment.")
        return 0
    else:
        print("✗ Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
