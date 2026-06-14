"""
QUANTUM-FORGE: System Health Monitor & Alerting
=================================================
Monitors key health metrics and triggers alerts when thresholds are breached.

Metrics tracked:
  - Pipeline latency (per-symbol processing time)
  - Model inference latency
  - Signal generation rate
  - Error rate per component  
  - Memory usage
  - WebSocket connection health
  - Database backend health
"""

import time
import logging
import threading
import psutil
from typing import Dict, List, Optional, Callable
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("HealthMonitor")


@dataclass
class HealthMetric:
    """A single health metric observation."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    unit: str = ""


@dataclass
class HealthThreshold:
    """Threshold configuration for a metric."""
    warn: float
    critical: float
    direction: str = "above"  # "above" or "below"


class SystemHealthMonitor:
    """
    Aggregates health metrics from all subsystems and triggers alerts.
    
    Runs a background thread that periodically checks health.
    """

    DEFAULT_THRESHOLDS = {
        "pipeline_latency_ms": HealthThreshold(warn=500, critical=2000, direction="above"),
        "model_latency_ms": HealthThreshold(warn=200, critical=1000, direction="above"),
        "error_rate_pct": HealthThreshold(warn=5.0, critical=20.0, direction="above"),
        "memory_pct": HealthThreshold(warn=80.0, critical=95.0, direction="above"),
        "signal_rate_per_min": HealthThreshold(warn=0.5, critical=0.1, direction="below"),
        "ws_reconnects": HealthThreshold(warn=3, critical=10, direction="above"),
    }

    def __init__(
        self,
        check_interval: float = 30.0,
        alert_callback: Optional[Callable] = None,
    ):
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)

        # Metric stores: {name: deque of HealthMetric}
        self._metrics: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Alert history
        self._alerts: deque = deque(maxlen=500)
        self._last_alert_time: Dict[str, float] = {}
        self._alert_cooldown = 300  # 5 min cooldown per metric

    def record(self, name: str, value: float, unit: str = ""):
        """Record a metric observation."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=500)
            self._metrics[name].append(HealthMetric(name=name, value=value, unit=unit))

    def start(self):
        """Start background health checking."""
        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        logger.info("HealthMonitor started")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("HealthMonitor stopped")

    def get_status(self) -> Dict:
        """Get current health status for all metrics."""
        status = {}
        with self._lock:
            for name, observations in self._metrics.items():
                if not observations:
                    continue
                recent = [m.value for m in list(observations)[-10:]]
                status[name] = {
                    "current": recent[-1],
                    "avg_10": sum(recent) / len(recent),
                    "min": min(recent),
                    "max": max(recent),
                    "count": len(observations),
                    "status": self._check_metric(name, recent[-1]),
                }

        # System-level metrics
        try:
            status["system"] = {
                "cpu_pct": psutil.cpu_percent(interval=0),
                "memory_pct": psutil.virtual_memory().percent,
                "disk_pct": psutil.disk_usage("/").percent if hasattr(psutil, "disk_usage") else 0,
            }
        except Exception:
            pass

        return status

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts."""
        return [
            {"time": a[0], "metric": a[1], "level": a[2], "message": a[3]}
            for a in list(self._alerts)[-limit:]
        ]

    def _check_loop(self):
        """Background health check loop."""
        while self._running:
            try:
                self._check_all()
            except Exception as e:
                logger.debug(f"Health check error: {e}")
            time.sleep(self.check_interval)

    def _check_all(self):
        """Check all metrics against thresholds."""
        # Record system metrics
        try:
            self.record("memory_pct", psutil.virtual_memory().percent, "%")
        except Exception:
            pass

        with self._lock:
            for name, observations in self._metrics.items():
                if not observations:
                    continue
                value = observations[-1].value
                level = self._check_metric(name, value)
                if level in ("WARN", "CRITICAL"):
                    self._fire_alert(name, level, value)

    def _check_metric(self, name: str, value: float) -> str:
        """Check a single metric. Returns: OK, WARN, CRITICAL."""
        threshold = self.thresholds.get(name)
        if not threshold:
            return "OK"

        if threshold.direction == "above":
            if value >= threshold.critical:
                return "CRITICAL"
            elif value >= threshold.warn:
                return "WARN"
        else:  # below
            if value <= threshold.critical:
                return "CRITICAL"
            elif value <= threshold.warn:
                return "WARN"

        return "OK"

    def _fire_alert(self, name: str, level: str, value: float):
        """Fire an alert (with cooldown)."""
        now = time.time()
        last = self._last_alert_time.get(name, 0)
        if now - last < self._alert_cooldown:
            return  # cooldown

        self._last_alert_time[name] = now
        msg = f"[{level}] {name} = {value:.2f} breached threshold"
        self._alerts.append((datetime.now().isoformat(), name, level, msg))
        logger.warning(msg)

        if self.alert_callback:
            try:
                self.alert_callback(level, name, value, msg)
            except Exception:
                pass
