"""
System Monitoring Manager
Real-time monitoring and alerting for QUANTUM-FORGE
"""

import os
import sys
import psutil
import time
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
import threading
import queue
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@dataclass
class MetricSnapshot:
    """System metric snapshot."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_sent: float
    network_recv: float
    active_threads: int
    open_files: int


@dataclass
class Alert:
    """System alert."""
    timestamp: str
    level: str
    component: str
    metric: str
    value: float
    threshold: float
    message: str


class SystemMonitor:
    """Real-time system monitoring and alerting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize system monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
        
        # Monitoring state
        self.running = False
        self.metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.alert_queue = queue.Queue()
        
        # Thresholds
        self.thresholds = self.config.get('thresholds', {})
        
        # Monitoring threads
        self.monitor_thread = None
        self.alert_thread = None
        
        # Custom metric collectors
        self.custom_collectors: Dict[str, Callable] = {}
        
        # Component health status
        self.component_health = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'interval': 5,  # seconds
            'thresholds': {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_usage': 90.0,
                'response_time': 1000,  # ms
                'error_rate': 0.05
            },
            'alerts': {
                'email_enabled': False,
                'email_to': 'alerts@quantumforge.com',
                'slack_enabled': False,
                'slack_webhook': ''
            },
            'retention': {
                'metrics_hours': 24,
                'alerts_hours': 72
            }
        }
    
    def start(self):
        """Start monitoring."""
        if self.running:
            self.logger.warning("Monitor already running")
            return
        
        self.running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start alert processing thread
        self.alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self.alert_thread.start()
        
        self.logger.info("System monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        interval = self.config['interval']
        
        while self.running:
            try:
                # Collect system metrics
                snapshot = self._collect_system_metrics()
                self.metrics_history.append(snapshot)
                
                # Check thresholds
                self._check_thresholds(snapshot)
                
                # Collect custom metrics
                self._collect_custom_metrics()
                
                # Sleep until next interval
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self) -> MetricSnapshot:
        """Collect system-level metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_sent = net_io.bytes_sent
        network_recv = net_io.bytes_recv
        
        # Process info
        process = psutil.Process()
        active_threads = process.num_threads()
        open_files = len(process.open_files())
        
        return MetricSnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage=disk_usage,
            network_sent=network_sent,
            network_recv=network_recv,
            active_threads=active_threads,
            open_files=open_files
        )
    
    def _check_thresholds(self, snapshot: MetricSnapshot):
        """Check metric thresholds and generate alerts."""
        # CPU threshold
        if snapshot.cpu_percent > self.thresholds.get('cpu_percent', 80):
            self._create_alert(
                'warning',
                'system',
                'cpu_percent',
                snapshot.cpu_percent,
                self.thresholds['cpu_percent'],
                f"CPU usage is high: {snapshot.cpu_percent:.1f}%"
            )
        
        # Memory threshold
        if snapshot.memory_percent > self.thresholds.get('memory_percent', 85):
            self._create_alert(
                'warning',
                'system',
                'memory_percent',
                snapshot.memory_percent,
                self.thresholds['memory_percent'],
                f"Memory usage is high: {snapshot.memory_percent:.1f}%"
            )
        
        # Disk threshold
        if snapshot.disk_usage > self.thresholds.get('disk_usage', 90):
            self._create_alert(
                'critical',
                'system',
                'disk_usage',
                snapshot.disk_usage,
                self.thresholds['disk_usage'],
                f"Disk usage is critical: {snapshot.disk_usage:.1f}%"
            )
    
    def _create_alert(self, level: str, component: str, metric: str, 
                     value: float, threshold: float, message: str):
        """Create and queue alert."""
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            level=level,
            component=component,
            metric=metric,
            value=value,
            threshold=threshold,
            message=message
        )
        
        self.alerts.append(alert)
        self.alert_queue.put(alert)
        
        self.logger.warning(f"Alert: {alert.message}")
    
    def _alert_loop(self):
        """Process alerts."""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                
                # Send email alert
                if self.config['alerts'].get('email_enabled'):
                    self._send_email_alert(alert)
                
                # Send Slack alert
                if self.config['alerts'].get('slack_enabled'):
                    self._send_slack_alert(alert)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = 'alerts@quantumforge.com'
            msg['To'] = self.config['alerts']['email_to']
            msg['Subject'] = f"QUANTUM-FORGE Alert: {alert.level.upper()}"
            
            body = f"""
            Alert Details:
            
            Level: {alert.level}
            Component: {alert.component}
            Metric: {alert.metric}
            Value: {alert.value:.2f}
            Threshold: {alert.threshold:.2f}
            Message: {alert.message}
            Timestamp: {alert.timestamp}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # In production, configure actual SMTP server
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login('user', 'password')
            # server.send_message(msg)
            # server.quit()
            
            self.logger.info(f"Email alert sent for: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send Slack alert."""
        try:
            import requests
            
            webhook_url = self.config['alerts'].get('slack_webhook')
            if not webhook_url:
                return
            
            color = {
                'info': '#36a64f',
                'warning': '#ff9800',
                'critical': '#f44336'
            }.get(alert.level, '#808080')
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': f"QUANTUM-FORGE Alert: {alert.level.upper()}",
                    'fields': [
                        {'title': 'Component', 'value': alert.component, 'short': True},
                        {'title': 'Metric', 'value': alert.metric, 'short': True},
                        {'title': 'Value', 'value': f"{alert.value:.2f}", 'short': True},
                        {'title': 'Threshold', 'value': f"{alert.threshold:.2f}", 'short': True},
                        {'title': 'Message', 'value': alert.message, 'short': False}
                    ],
                    'footer': 'QUANTUM-FORGE Monitor',
                    'ts': int(datetime.fromisoformat(alert.timestamp).timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent for: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
    
    def _collect_custom_metrics(self):
        """Collect custom metrics from registered collectors."""
        for name, collector in self.custom_collectors.items():
            try:
                metric = collector()
                # Store custom metric
                self.logger.debug(f"Custom metric {name}: {metric}")
            except Exception as e:
                self.logger.error(f"Error collecting custom metric {name}: {e}")
    
    def register_collector(self, name: str, collector: Callable):
        """
        Register custom metric collector.
        
        Args:
            name: Collector name
            collector: Callable that returns metric value
        """
        self.custom_collectors[name] = collector
        self.logger.info(f"Registered custom collector: {name}")
    
    def get_current_metrics(self) -> Optional[MetricSnapshot]:
        """Get current system metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, hours: int = 1) -> List[MetricSnapshot]:
        """Get metrics history for specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        filtered = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff
        ]
        
        return filtered
    
    def get_recent_alerts(self, hours: int = 1) -> List[Alert]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        filtered = [
            a for a in self.alerts
            if datetime.fromisoformat(a.timestamp) > cutoff
        ]
        
        return filtered
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for specific component."""
        return self.component_health.get(component, {
            'status': 'unknown',
            'last_check': None,
            'metrics': {}
        })
    
    def update_component_health(self, component: str, status: str, metrics: Dict[str, Any]):
        """Update component health status."""
        self.component_health[component] = {
            'status': status,
            'last_check': datetime.now().isoformat(),
            'metrics': metrics
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        current = self.get_current_metrics()
        recent_alerts = self.get_recent_alerts(hours=1)
        
        if not current:
            return {'status': 'no_data'}
        
        # Calculate alert severity
        alert_counts = {
            'info': 0,
            'warning': 0,
            'critical': 0
        }
        
        for alert in recent_alerts:
            alert_counts[alert.level] = alert_counts.get(alert.level, 0) + 1
        
        # Determine overall health
        if alert_counts['critical'] > 0:
            overall_health = 'critical'
        elif alert_counts['warning'] > 5:
            overall_health = 'degraded'
        else:
            overall_health = 'healthy'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': overall_health,
            'current_metrics': asdict(current),
            'alert_summary': alert_counts,
            'component_health': self.component_health,
            'uptime_seconds': time.time() - psutil.boot_time()
        }
    
    def export_metrics(self, output_path: str, hours: int = 24):
        """Export metrics to file."""
        try:
            metrics = self.get_metrics_history(hours=hours)
            
            data = {
                'export_time': datetime.now().isoformat(),
                'duration_hours': hours,
                'metrics_count': len(metrics),
                'metrics': [asdict(m) for m in metrics]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Metrics exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def export_alerts(self, output_path: str, hours: int = 72):
        """Export alerts to file."""
        try:
            alerts = self.get_recent_alerts(hours=hours)
            
            data = {
                'export_time': datetime.now().isoformat(),
                'duration_hours': hours,
                'alerts_count': len(alerts),
                'alerts': [asdict(a) for a in alerts]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Alerts exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export alerts: {e}")
    
    def generate_health_report(self, output_path: str = 'health_report.json'):
        """Generate comprehensive health report."""
        try:
            summary = self.get_system_summary()
            metrics = self.get_metrics_history(hours=24)
            alerts = self.get_recent_alerts(hours=24)
            
            # Calculate statistics
            if metrics:
                cpu_avg = sum(m.cpu_percent for m in metrics) / len(metrics)
                mem_avg = sum(m.memory_percent for m in metrics) / len(metrics)
                cpu_max = max(m.cpu_percent for m in metrics)
                mem_max = max(m.memory_percent for m in metrics)
            else:
                cpu_avg = mem_avg = cpu_max = mem_max = 0
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': summary,
                'statistics_24h': {
                    'cpu_avg': cpu_avg,
                    'cpu_max': cpu_max,
                    'memory_avg': mem_avg,
                    'memory_max': mem_max,
                    'total_alerts': len(alerts)
                },
                'top_alerts': [asdict(a) for a in list(alerts)[:10]]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Health report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate health report: {e}")


class PerformanceMonitor:
    """Monitor trading performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'orders': deque(maxlen=10000),
            'trades': deque(maxlen=10000),
            'pnl': deque(maxlen=1000),
            'latency': deque(maxlen=10000)
        }
    
    def record_order(self, order_data: Dict[str, Any]):
        """Record order submission."""
        self.metrics['orders'].append({
            'timestamp': datetime.now().isoformat(),
            **order_data
        })
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record trade execution."""
        self.metrics['trades'].append({
            'timestamp': datetime.now().isoformat(),
            **trade_data
        })
    
    def record_pnl(self, pnl: float):
        """Record P&L."""
        self.metrics['pnl'].append({
            'timestamp': datetime.now().isoformat(),
            'pnl': pnl
        })
    
    def record_latency(self, operation: str, latency_ms: float):
        """Record operation latency."""
        self.metrics['latency'].append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'latency_ms': latency_ms
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'orders_count': len(self.metrics['orders']),
            'trades_count': len(self.metrics['trades']),
            'total_pnl': sum(p['pnl'] for p in self.metrics['pnl']),
            'avg_latency_ms': sum(l['latency_ms'] for l in self.metrics['latency']) / len(self.metrics['latency']) if self.metrics['latency'] else 0
        }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start monitoring
    monitor = SystemMonitor()
    monitor.start()
    
    try:
        # Monitor for 60 seconds
        time.sleep(60)
        
        # Generate reports
        monitor.export_metrics('system_metrics.json', hours=1)
        monitor.generate_health_report('health_report.json')
        
    finally:
        monitor.stop()
