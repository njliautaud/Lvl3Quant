"""
Safety Mechanisms Test Suite
=============================

Tests all failure modes and protection layers.
"""

import time
from pathlib import Path
from live_trading.safety import (
    ModelHealthCheck,
    OrderValidator,
    RiskMonitor,
    CircuitBreaker,
    KillSwitch,
    SafetyManager,
    Order,
    Position,
)


class MockPrediction:
    """Mock prediction for testing"""
    def __init__(self, value: float, latency_ms: float):
        self.value = value
        self.latency_ms = latency_ms


def test_model_health_check():
    """Test model health validation"""
    checker = ModelHealthCheck(max_prediction_value=5.0, max_latency_ms=30.0)

    # Valid prediction
    pred = MockPrediction(value=2.0, latency_ms=10.0)
    valid, msg = checker.validate_prediction(pred)
    assert valid, msg

    # Out of range prediction
    pred = MockPrediction(value=15.0, latency_ms=10.0)
    valid, msg = checker.validate_prediction(pred)
    assert not valid
    assert "out of range" in msg.lower()

    # High latency
    pred = MockPrediction(value=2.0, latency_ms=100.0)
    valid, msg = checker.validate_prediction(pred)
    assert not valid
    assert "latency" in msg.lower()

    # NaN prediction
    pred = MockPrediction(value=float('nan'), latency_ms=10.0)
    valid, msg = checker.validate_prediction(pred)
    assert not valid
    assert "nan" in msg.lower() or "inf" in msg.lower()


def test_order_validator():
    """Test order validation"""
    from datetime import time as dtime
    # Set trading hours to 24/7 for testing
    validator = OrderValidator(
        max_position_per_card=1,
        max_total_position=3,
        max_daily_trades=10,
        max_daily_loss=100.0,
        trading_hours_start=dtime(0, 0),
        trading_hours_end=dtime(23, 59),
    )

    # Valid order
    order = Order(
        card_name="test_card",
        side="BUY",
        size=1,
        symbol="ES",
        price=5000.0,
        order_type="MARKET",
        timestamp=time.time()
    )
    valid, msg = validator.validate_order(order, {})
    assert valid, msg

    # Duplicate order (same card, side, within 1 second)
    order2 = Order(
        card_name="test_card",
        side="BUY",
        size=1,
        symbol="ES",
        price=5000.1,
        order_type="MARKET",
        timestamp=time.time()
    )
    valid, msg = validator.validate_order(order2, {})
    assert not valid
    assert "duplicate" in msg.lower()

    # Invalid symbol
    order3 = Order(
        card_name="test_card",
        side="BUY",
        size=1,
        symbol="FAKE",
        price=5000.0,
        order_type="MARKET",
        timestamp=time.time() + 2
    )
    valid, msg = validator.validate_order(order3, {})
    assert not valid
    assert "whitelist" in msg.lower() or "symbol" in msg.lower()


def test_position_limits():
    """Test position limit enforcement"""
    validator = OrderValidator(
        max_position_per_card=2,
        max_total_position=5,
    )

    # Create existing position
    existing_pos = Position(
        card_name="card1",
        side="LONG",
        size=1,
        entry_price=5000.0,
        entry_time=time.time(),
        unrealized_pnl=0.0
    )

    # Order that would exceed card limit
    order = Order(
        card_name="card1",
        side="BUY",
        size=2,
        symbol="ES",
        price=5001.0,
        order_type="MARKET",
        timestamp=time.time()
    )
    valid, msg = validator.validate_order(order, {"card1": existing_pos})
    assert not valid
    assert "limit" in msg.lower()


def test_risk_monitor():
    """Test risk monitoring"""
    monitor = RiskMonitor(
        max_position_hold_seconds=60,
        max_unrealized_loss_per_position=50.0,
    )

    # Position held too long
    old_pos = Position(
        card_name="card1",
        side="LONG",
        size=1,
        entry_price=5000.0,
        entry_time=time.time() - 100,  # 100 seconds ago
        unrealized_pnl=-10.0
    )

    to_close = monitor.check_positions({"card1": old_pos})
    assert len(to_close) == 1
    assert "hold time" in to_close[0][1].lower()

    # Position with large loss
    losing_pos = Position(
        card_name="card2",
        side="LONG",
        size=1,
        entry_price=5000.0,
        entry_time=time.time(),
        unrealized_pnl=-50.0  # -50 ticks = $625
    )

    to_close = monitor.check_positions({"card2": losing_pos})
    assert len(to_close) == 1
    assert "loss" in to_close[0][1].lower()


def test_circuit_breaker():
    """Test circuit breaker triggers"""
    breaker = CircuitBreaker(
        max_consecutive_losses=3,
        max_hourly_trades=5,
    )

    # Record consecutive losses
    breaker.record_trade(pnl=-1.0, entry_price=5000.0, fill_price=5000.5)
    breaker.record_trade(pnl=-1.0, entry_price=5001.0, fill_price=5001.5)
    should_halt, reason = breaker.check_circuit_breakers()
    assert not should_halt  # Not yet

    breaker.record_trade(pnl=-1.0, entry_price=5002.0, fill_price=5002.5)
    should_halt, reason = breaker.check_circuit_breakers()
    assert should_halt
    assert "consecutive" in reason.lower()

    # Reset and test trade rate
    breaker.reset()
    for i in range(6):
        breaker.record_trade(pnl=1.0, entry_price=5000.0, fill_price=5000.0)

    should_halt, reason = breaker.check_circuit_breakers()
    assert should_halt
    assert "rate" in reason.lower()


def test_kill_switch():
    """Test kill switch activation"""
    kill_file = Path("/tmp/test_kill_switch")
    if kill_file.exists():
        kill_file.unlink()

    switch = KillSwitch(kill_file_path=str(kill_file))

    # Not activated
    active, reason = switch.check()
    assert not active

    # Activate
    switch.activate("Test activation")
    active, reason = switch.check()
    assert active
    assert "test activation" in reason.lower()

    # Deactivate
    switch.deactivate()
    active, reason = switch.check()
    assert not active


def test_safety_manager_integration():
    """Test SafetyManager integrates all components"""
    config = {
        'max_card_position': 1,
        'max_total_position': 3,
        'max_daily_trades': 100,
        'max_daily_loss_dollars': 500.0,
    }

    manager = SafetyManager(config)

    # Test prediction validation
    pred = MockPrediction(value=2.0, latency_ms=10.0)
    valid, msg = manager.validate_prediction(pred)
    assert valid

    # Test order validation
    order = Order(
        card_name="test",
        side="BUY",
        size=1,
        symbol="ES",
        price=5000.0,
        order_type="MARKET",
        timestamp=time.time()
    )
    valid, msg = manager.validate_order(order, {})
    assert valid

    # Test halt check
    should_halt, reason = manager.should_halt()
    assert not should_halt


if __name__ == "__main__":
    # Run tests
    print("Running safety mechanism tests...")

    test_model_health_check()
    print("✓ Model health check")

    test_order_validator()
    print("✓ Order validator")

    test_position_limits()
    print("✓ Position limits")

    test_risk_monitor()
    print("✓ Risk monitor")

    test_circuit_breaker()
    print("✓ Circuit breaker")

    test_kill_switch()
    print("✓ Kill switch")

    test_safety_manager_integration()
    print("✓ Safety manager integration")

    print("\n✅ All safety tests passed!")
