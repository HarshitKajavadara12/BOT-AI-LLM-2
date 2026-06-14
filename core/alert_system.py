"""
QUANTUM-FORGE: Alerting System
================================
P3 — Real-time alerts via Telegram and email for:
  - Trade executions
  - Risk breaches (position limits, drawdown)
  - Regime changes
  - System errors
  - Portfolio milestones (new highs, drawdown recovery)

Configuration via environment variables:
  TELEGRAM_BOT_TOKEN - Telegram Bot API token
  TELEGRAM_CHAT_ID   - Telegram chat/channel ID
  ALERT_EMAIL_FROM    - Sender email
  ALERT_EMAIL_TO      - Recipient email
  ALERT_SMTP_HOST     - SMTP server host
  ALERT_SMTP_PORT     - SMTP server port
  ALERT_SMTP_USER     - SMTP username
  ALERT_SMTP_PASSWORD - SMTP password
"""

import os
import json
import logging
import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from enum import Enum

logger = logging.getLogger("AlertSystem")


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertType(Enum):
    TRADE = "TRADE"
    RISK = "RISK"
    REGIME = "REGIME"
    PORTFOLIO = "PORTFOLIO"
    SYSTEM = "SYSTEM"


class AlertSystem:
    """
    Multi-channel alert system for the trading platform.
    
    Supports Telegram and email. Falls back gracefully if
    channels are not configured.
    """
    
    # Rate limiting
    MAX_ALERTS_PER_MINUTE = 10
    MIN_INTERVAL_SECONDS = 5  # Between same-type alerts
    
    def __init__(self):
        # Telegram config
        self.telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.telegram_enabled = bool(self.telegram_token and self.telegram_chat_id)
        
        # Email config
        self.email_from = os.environ.get('ALERT_EMAIL_FROM')
        self.email_to = os.environ.get('ALERT_EMAIL_TO')
        self.smtp_host = os.environ.get('ALERT_SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('ALERT_SMTP_PORT', '587'))
        self.smtp_user = os.environ.get('ALERT_SMTP_USER')
        self.smtp_password = os.environ.get('ALERT_SMTP_PASSWORD')
        self.email_enabled = bool(self.email_from and self.email_to and self.smtp_user)
        
        # Rate limiting
        self._alert_timestamps: deque = deque(maxlen=100)
        self._last_alert_by_type: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Alert history
        self.alert_history: List[Dict] = []
        
        # Background queue for async delivery
        self._queue: deque = deque(maxlen=500)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        channels = []
        if self.telegram_enabled:
            channels.append("Telegram")
        if self.email_enabled:
            channels.append("Email")
        
        if channels:
            logger.info(f"[OK] Alert System ({', '.join(channels)})")
        else:
            logger.info("[OK] Alert System (no channels configured — alerts logged only)")
    
    def start(self):
        """Start the background alert delivery thread."""
        self._running = True
        self._thread = threading.Thread(target=self._delivery_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the alert delivery thread."""
        self._running = False
    
    def _delivery_loop(self):
        """Background loop that delivers queued alerts."""
        while self._running:
            try:
                if self._queue:
                    alert = self._queue.popleft()
                    self._deliver(alert)
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Alert delivery error: {e}")
                time.sleep(5)
    
    # === High-level alert methods ===
    
    def trade_alert(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        algo: str = "MARKET",
        fee: float = 0.0,
    ):
        """Alert on trade execution."""
        if side == "BUY":
            emoji = "🟢"
            msg = f"{emoji} BUY {quantity:.6f} {symbol} @ ${price:,.2f}"
        else:
            emoji = "🔴"
            pnl_str = f" | P&L: ${pnl:+,.2f}" if pnl is not None else ""
            msg = f"{emoji} SELL {quantity:.6f} {symbol} @ ${price:,.2f}{pnl_str}"
        
        msg += f"\n  Algo: {algo} | Fee: ${fee:.2f}"
        
        self._send(
            level=AlertLevel.INFO,
            alert_type=AlertType.TRADE,
            title=f"Trade: {side} {symbol}",
            message=msg,
        )
    
    def risk_alert(self, message: str, level: AlertLevel = AlertLevel.WARNING):
        """Alert on risk events."""
        emoji = "⚠️" if level == AlertLevel.WARNING else "🚨"
        self._send(
            level=level,
            alert_type=AlertType.RISK,
            title=f"{emoji} Risk Alert",
            message=message,
        )
    
    def regime_change_alert(self, old_regime: str, new_regime: str, confidence: float):
        """Alert on market regime changes."""
        self._send(
            level=AlertLevel.INFO,
            alert_type=AlertType.REGIME,
            title="📊 Regime Change",
            message=f"Market regime: {old_regime} → {new_regime} (confidence: {confidence:.0%})",
        )
    
    def portfolio_alert(self, message: str, level: AlertLevel = AlertLevel.INFO):
        """Alert on portfolio milestones."""
        self._send(
            level=level,
            alert_type=AlertType.PORTFOLIO,
            title="💼 Portfolio",
            message=message,
        )
    
    def system_alert(self, message: str, level: AlertLevel = AlertLevel.WARNING):
        """Alert on system events (errors, restarts, etc.)."""
        emoji = "⚙️" if level == AlertLevel.INFO else "🔧"
        self._send(
            level=level,
            alert_type=AlertType.SYSTEM,
            title=f"{emoji} System",
            message=message,
        )
    
    # === Core delivery ===
    
    def _send(self, level: AlertLevel, alert_type: AlertType, title: str, message: str):
        """Queue an alert for delivery."""
        now = time.time()
        
        with self._lock:
            # Rate limit check
            recent = [t for t in self._alert_timestamps if now - t < 60]
            if len(recent) >= self.MAX_ALERTS_PER_MINUTE:
                logger.debug(f"Alert rate-limited: {title}")
                return
            
            # Same-type cooldown
            type_key = f"{alert_type.value}_{level.value}"
            last_time = self._last_alert_by_type.get(type_key, 0)
            if now - last_time < self.MIN_INTERVAL_SECONDS:
                logger.debug(f"Alert throttled: {title}")
                return
            
            self._alert_timestamps.append(now)
            self._last_alert_by_type[type_key] = now
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'type': alert_type.value,
            'title': title,
            'message': message,
        }
        
        self.alert_history.append(alert)
        
        # Log locally always
        log_msg = f"[ALERT:{level.value}] {title}: {message}"
        if level in (AlertLevel.CRITICAL, AlertLevel.EMERGENCY):
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        # Queue for async delivery
        if self._running and (self.telegram_enabled or self.email_enabled):
            self._queue.append(alert)
    
    def _deliver(self, alert: Dict):
        """Deliver an alert through all configured channels."""
        level = alert.get('level', 'INFO')
        
        # Telegram: send all alerts
        if self.telegram_enabled:
            self._send_telegram(alert)
        
        # Email: only WARNING+ alerts
        if self.email_enabled and level in ('WARNING', 'CRITICAL', 'EMERGENCY'):
            self._send_email(alert)
    
    def _send_telegram(self, alert: Dict):
        """Send alert via Telegram Bot API."""
        try:
            import requests
            
            text = (
                f"*{alert['title']}*\n"
                f"_{alert['timestamp']}_\n\n"
                f"{alert['message']}"
            )
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': text,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True,
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Telegram send failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    def _send_email(self, alert: Dict):
        """Send alert via email (SMTP)."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = f"[Quantum-Forge] {alert['title']}"
            
            body = (
                f"Alert Level: {alert['level']}\n"
                f"Type: {alert['type']}\n"
                f"Time: {alert['timestamp']}\n\n"
                f"{alert['message']}\n\n"
                f"— Quantum-Forge Trading System"
            )
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
        except Exception as e:
            logger.error(f"Email error: {e}")
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        if not self.alert_history:
            return {'total': 0}
        
        level_counts = {}
        type_counts = {}
        for a in self.alert_history:
            level_counts[a['level']] = level_counts.get(a['level'], 0) + 1
            type_counts[a['type']] = type_counts.get(a['type'], 0) + 1
        
        return {
            'total': len(self.alert_history),
            'by_level': level_counts,
            'by_type': type_counts,
            'channels': {
                'telegram': self.telegram_enabled,
                'email': self.email_enabled,
            },
        }
