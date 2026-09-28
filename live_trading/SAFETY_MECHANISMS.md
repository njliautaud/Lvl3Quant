# Live Trading Safety Mechanisms — Finance-Grade Robustness

**CRITICAL**: This is real money. Every failure mode must be anticipated and handled.

---

## Core Principles

1. **Fail-Safe Defaults**: System defaults to NO TRADING when uncertain
2. **Multiple Validation Layers**: Every order checked multiple times
3. **Kill Switch Always Available**: Human override at any moment
4. **Audit Everything**: Complete trail of every decision
5. **Test Everything**: No code goes live without testing

---

## Layer 1: Pre-Trade Validation

### Model Health Checks
```python
class ModelHealthCheck:
    """Verify model is producing valid predictions"""
    
    def validate_prediction(self, pred: Prediction) -> bool:
        # 1. Prediction in reasonable range?
        if abs(pred.value) > 10.0:  # >10 ticks is suspicious
            logger.error(f"Prediction out of range: {pred.value}")
            return False
        
        # 2. Model latency acceptable?
        if pred.latency_ms > 50:  # Too slow
            logger.warning(f"Model latency high: {pred.latency_ms}ms")
            return False
        
        # 3. Not NaN or Inf?
        if not np.isfinite(pred.value):
            logger.error("Prediction is NaN or Inf")
            return False
        
        # 4. Recent predictions not all same? (model frozen?)
        if self._check_frozen_model(pred):
            logger.error("Model appears frozen")
            return False
        
        return True
```

### Order Validation
```python
class OrderValidator:
    """Validate every order before submission"""
    
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        # 1. Position limits
        if self._would_exceed_position_limit(order):
            return False, "Position limit exceeded"
        
        # 2. Price sanity check
        if not self._price_reasonable(order.price):
            return False, "Price unreasonable"
        
        # 3. Duplicate order detection
        if self._is_duplicate(order):
            return False, "Duplicate order detected"
        
        # 4. Daily trade limit
        if self._exceeded_daily_trades():
            return False, "Daily trade limit reached"
        
        # 5. Daily loss limit
        if self._exceeded_daily_loss():
            return False, "Daily loss limit reached"
        
        # 6. Market hours check
        if not self._in_trading_hours():
            return False, "Outside trading hours"
        
        # 7. Symbol whitelist
        if order.symbol not in self.allowed_symbols:
            return False, f"Symbol {order.symbol} not whitelisted"
        
        return True, "OK"
```

---

## Layer 2: Position Management

### Real-Time Risk Monitoring
```python
class RiskMonitor:
    """Monitor risk in real-time, kill positions if needed"""
    
    def __init__(self):
        self.max_drawdown_pct = 0.02  # 2% of account
        self.max_position_time = 3600  # 1 hour max hold
        self.max_unrealized_loss = 500  # $500 per position
    
    def check_positions(self, positions: Dict):
        for card_name, pos in positions.items():
            # 1. Position held too long?
            if pos.hold_time > self.max_position_time:
                self.force_close(pos, "Max hold time exceeded")
            
            # 2. Unrealized loss too large?
            if pos.unrealized_pnl < -self.max_unrealized_loss:
                self.force_close(pos, "Max loss exceeded")
            
            # 3. Account drawdown limit?
            if self.account_drawdown() > self.max_drawdown_pct:
                self.close_all("Account drawdown limit")
```

### Circuit Breakers
```python
class CircuitBreaker:
    """Automatic trading halts on anomalous conditions"""
    
    def __init__(self):
        self.max_consecutive_losses = 5
        self.max_hourly_trades = 50
        self.min_win_rate_threshold = 0.35  # Below 35% stop
    
    def check_circuit_breakers(self):
        # 1. Too many consecutive losses?
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.halt_trading("Consecutive losses")
        
        # 2. Trading too fast?
        if self.trades_last_hour() > self.max_hourly_trades:
            self.halt_trading("Trade rate too high")
        
        # 3. Win rate collapsed?
        if self.recent_win_rate() < self.min_win_rate_threshold:
            self.halt_trading("Win rate below threshold")
        
        # 4. Fill quality degraded?
        if self.avg_slippage() > self.max_slippage:
            self.halt_trading("Excessive slippage")
```

---

## Layer 3: Data Integrity

### Leakage Prevention
```python
class LeakageAuditor:
    """Prevent look-ahead bias and data leakage"""
    
    def audit_features(self, features: np.ndarray, timestamp: int):
        # 1. No future timestamps in features?
        if hasattr(features, 'timestamp'):
            if features.timestamp > timestamp:
                raise LeakageError("Future data in features")
        
        # 2. Feature statistics match expected?
        if not self._stats_match_train(features):
            logger.warning("Feature distribution shift detected")
        
        # 3. No NaN/Inf propagation?
        if not np.isfinite(features).all():
            raise ValueError("NaN/Inf in features")
```

### Model Staleness Check
```python
class ModelFreshness:
    """Ensure model hasn't decayed"""
    
    def __init__(self):
        self.max_days_old = 30
        self.min_acceptable_ic = 0.05
    
    def check_model_fresh(self, model_meta: ModelMetadata):
        # 1. Model too old?
        age_days = (datetime.now() - model_meta.train_date).days
        if age_days > self.max_days_old:
            logger.warning(f"Model {age_days} days old")
        
        # 2. Recent performance acceptable?
        recent_ic = self.calculate_recent_ic(window=1000)
        if recent_ic < self.min_acceptable_ic:
            logger.error(f"Model IC degraded: {recent_ic}")
            return False
        
        return True
```

---

## Layer 4: Kill Switches

### Manual Override
```python
class KillSwitch:
    """Emergency stop mechanism"""
    
    def __init__(self):
        self.enabled = True
        self.kill_file = Path("/tmp/trading_kill_switch")
    
    def check_kill_switch(self):
        # 1. Kill file exists?
        if self.kill_file.exists():
            self.emergency_stop("Kill switch activated")
        
        # 2. Discord command?
        if self.discord_kill_command_received():
            self.emergency_stop("Discord kill command")
        
        # 3. Keyboard interrupt?
        # Handled by signal handlers
    
    def emergency_stop(self, reason: str):
        logger.critical(f"EMERGENCY STOP: {reason}")
        # 1. Cancel all open orders
        self.cancel_all_orders()
        # 2. Close all positions (market orders)
        self.close_all_positions()
        # 3. Disable all cards
        self.disable_all_cards()
        # 4. Send alert
        self.send_discord_alert(f"🚨 EMERGENCY STOP: {reason}")
        # 5. Exit gracefully
        sys.exit(0)
```

### Automated Halt Conditions
```python
# In main loop:
def main_loop():
    while running:
        # Check kill switches FIRST
        if kill_switch.check_kill_switch():
            break
        
        # Check circuit breakers
        if circuit_breaker.should_halt():
            kill_switch.emergency_stop("Circuit breaker triggered")
        
        # Check risk limits
        if risk_monitor.risk_too_high():
            kill_switch.emergency_stop("Risk limit exceeded")
        
        # ... normal trading logic ...
```

---

## Layer 5: Monitoring & Alerts

### Real-Time Alerts
```python
class AlertSystem:
    """Send immediate alerts on critical events"""
    
    def __init__(self, discord_webhook: str):
        self.webhook = discord_webhook
        self.alert_levels = {
            'CRITICAL': self.send_immediately,
            'WARNING': self.send_if_repeated,
            'INFO': self.batch_and_send
        }
    
    def alert(self, level: str, message: str):
        if level == 'CRITICAL':
            # 1. Discord DM
            self.send_discord_dm(message)
            # 2. Webhook
            self.send_webhook(message)
            # 3. Log
            logger.critical(message)
            # 4. Email (optional)
            if self.email_enabled:
                self.send_email(message)
```

### Watchdog Timer
```python
class Watchdog:
    """Ensure system is responsive"""
    
    def __init__(self):
        self.last_heartbeat = time.time()
        self.timeout = 60  # 60 seconds
    
    def heartbeat(self):
        self.last_heartbeat = time.time()
    
    def check_alive(self):
        if time.time() - self.last_heartbeat > self.timeout:
            logger.error("System appears frozen")
            # Send alert and attempt recovery
            self.attempt_recovery()
```

---

## Layer 6: Audit Trail

### Complete Logging
```python
class AuditLogger:
    """Log every decision for post-trade analysis"""
    
    def log_prediction(self, pred: Prediction):
        self.append_to_audit_log({
            'type': 'prediction',
            'timestamp': time.time(),
            'model': pred.model_name,
            'value': pred.value,
            'latency_ms': pred.latency_ms,
            'features_hash': pred.features_hash
        })
    
    def log_order(self, order: Order, reason: str):
        self.append_to_audit_log({
            'type': 'order',
            'timestamp': time.time(),
            'card': order.card_name,
            'side': order.side,
            'size': order.size,
            'price': order.price,
            'reason': reason
        })
    
    def log_rejection(self, order: Order, reason: str):
        self.append_to_audit_log({
            'type': 'rejection',
            'timestamp': time.time(),
            'order': order.to_dict(),
            'reason': reason
        })
```

---

## Implementation Checklist

### Week 1 (Conservative Start)
- [ ] Implement all validation layers
- [ ] Test kill switch (dry-run)
- [ ] Setup Discord alerts
- [ ] Configure conservative limits:
  - [ ] Max 1 contract position
  - [ ] $500 daily loss limit
  - [ ] 200 trade/day limit
- [ ] Deploy with SINGLE model only
- [ ] RTH only (9:30-4:00 ET)

### Week 2 (Expand Cautiously)
- [ ] Increase to 2 contracts if Week 1 successful
- [ ] Add second model (if validated)
- [ ] Expand to 2 cards
- [ ] Monitor slippage vs backtest

### Week 3+ (Full Deployment)
- [ ] Multiple models
- [ ] Multiple cards
- [ ] Overnight sessions (if applicable)
- [ ] Auto-rebalancing

---

## Testing Requirements

### Pre-Live Testing
1. **Kill Switch Test**: Verify emergency stop works
2. **Position Limit Test**: Confirm limits enforced
3. **Loss Limit Test**: Verify trading halts at limit
4. **Duplicate Order Test**: Confirm detection works
5. **Leakage Audit**: Run on all models before deployment
6. **Stress Test**: 30-day replay with all safety checks enabled

### Live Testing (Day 1)
1. **First Trade**: Monitor manually, verify all checks passed
2. **First 10 Trades**: Close monitoring
3. **First 50 Trades**: Periodic checks
4. **After 100 Trades**: Can relax monitoring if all good

---

## Recovery Procedures

### If Model Fails
1. Kill switch activated
2. Close all positions
3. Switch to backup model (if available)
4. Alert sent
5. Manual review required before restart

### If Exchange Connection Drops
1. Assume all pending orders filled (worst case)
2. Attempt to reconnect
3. Query positions from exchange
4. Reconcile with internal state
5. If can't reconcile, halt trading

### If System Crashes
1. On restart, query exchange for open positions
2. Reconcile with last known state
3. Close any unexpected positions
4. Resume only after manual review

---

**Remember**: In finance, paranoia is a feature, not a bug. Better to miss trades than to blow up the account.
