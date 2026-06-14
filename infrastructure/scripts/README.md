# QUANTUM-FORGE Infrastructure Scripts

Comprehensive automation suite for deployment, monitoring, maintenance, and backup of the QUANTUM-FORGE trading system.

##   Components

### 1. Deployment Manager (`deployment.py`)
Automated system deployment and initialization.

**Features:**
- Environment validation and dependency checking
- Database schema initialization
- Redis cache setup
- Component deployment orchestration
- Worker process management
- Deployment verification and rollback
- Comprehensive deployment reporting

**Usage:**
```python
from infrastructure.scripts.deployment import SystemDeployer

# Initialize deployer
deployer = SystemDeployer('deployment_config.yaml')

# Execute deployment
summary = deployer.deploy()

# Generate report
deployer.generate_deployment_report(summary)

# Rollback if needed
# deployer.rollback()
```

**CLI Usage:**
```bash
python infrastructure/scripts/deployment.py
```

### 2. System Monitor (`monitoring.py`)
Real-time system monitoring and alerting.

**Features:**
- System resource monitoring (CPU, memory, disk, network)
- Custom metric collection
- Threshold-based alerting
- Email and Slack notifications
- Component health tracking
- Performance metrics collection
- Comprehensive health reporting

**Usage:**
```python
from infrastructure.scripts.monitoring import SystemMonitor

# Initialize monitor
monitor = SystemMonitor()

# Start monitoring
monitor.start()

# Get current status
summary = monitor.get_system_summary()

# Export metrics
monitor.export_metrics('metrics.json', hours=24)
monitor.generate_health_report('health.json')

# Stop monitoring
monitor.stop()
```

**Custom Metric Collection:**
```python
def collect_order_latency():
    # Your metric collection logic
    return average_latency

# Register custom collector
monitor.register_collector('order_latency', collect_order_latency)
```

### 3. Maintenance Manager (`maintenance.py`)
Automated system maintenance tasks.

**Features:**
- Database vacuum and optimization
- Cache cleanup
- Log rotation and compression
- Disk space management
- Data archival
- Health checks
- Data integrity verification

**Usage:**
```python
from infrastructure.scripts.maintenance import MaintenanceManager

# Initialize manager
manager = MaintenanceManager()

# Run specific tasks
manager.cleanup_cache()
manager.rotate_logs()
manager.vacuum_database()

# Run health checks
health = manager.run_health_checks()

# Generate maintenance report
manager.generate_maintenance_report()

# Start scheduler for automatic maintenance
# manager.start()  # Runs tasks on schedule
```

**Scheduled Tasks:**
- **Hourly**: Cache cleanup, health checks
- **Daily**: Database vacuum, log rotation, data archival
- **Weekly**: Disk cleanup, index optimization

### 4. Backup Manager (`backup.py`)
Comprehensive backup and recovery system.

**Features:**
- Full system backups
- Component-level backup/restore
- Compression and encryption support
- Checksum verification
- Automated retention management
- Backup integrity verification
- Comprehensive backup reporting

**Usage:**
```python
from infrastructure.scripts.backup import BackupManager

# Initialize manager
manager = BackupManager()

# Create full backup
backup_info = manager.create_full_backup()

# List backups
backups = manager.list_backups()

# Restore from backup
manager.restore_backup(backup_id='backup_20240101_120000_abc123')

# Verify backup integrity
is_valid = manager.verify_backup(backup_id)

# Cleanup old backups
manager.cleanup_old_backups()

# Generate report
manager.generate_backup_report()
```

**Component-Specific Backup:**
```python
# Backup only specific components
backup_info = manager.create_full_backup(
    components=['database', 'config', 'models']
)

# Restore specific components
manager.restore_backup(
    backup_id='backup_xyz',
    components=['config']
)
```

##   Configuration

### Deployment Configuration (`deployment_config.yaml`)
```yaml
environment: production

components:
  core: true
  market_microstructure: true
  risk_engine: true
  execution: true
  ai_ml: true
  analytics: true
  dashboards: true
  data_systems: true

database:
  type: postgresql
  host: localhost
  port: 5432
  name: quantum_forge
  user: postgres

redis:
  host: localhost
  port: 6379
  db: 0

workers:
  data_ingestion: 4
  order_execution: 8
  risk_monitoring: 2
  analytics: 4

monitoring:
  metrics_port: 9090
  logging_level: INFO
  alert_email: alerts@quantumforge.com
```

### Monitoring Configuration
```python
config = {
    'interval': 5,  # seconds
    'thresholds': {
        'cpu_percent': 80.0,
        'memory_percent': 85.0,
        'disk_usage': 90.0,
        'response_time': 1000,  # ms
        'error_rate': 0.05
    },
    'alerts': {
        'email_enabled': True,
        'email_to': 'alerts@quantumforge.com',
        'slack_enabled': True,
        'slack_webhook': 'https://hooks.slack.com/...'
    }
}
```

### Maintenance Configuration
```python
config = {
    'log_retention_days': 30,
    'data_retention_days': 90,
    'backup_retention_days': 7,
    'database_vacuum_schedule': 'daily',
    'cache_cleanup_schedule': 'hourly',
    'log_rotation_schedule': 'daily'
}
```

### Backup Configuration
```python
config = {
    'backup_path': 'backups',
    'retention_days': 30,
    'compression': True,
    'encryption': False,
    'components': {
        'database': True,
        'config': True,
        'logs': True,
        'data': True,
        'models': True
    }
}
```

##   Quick Start

### Initial Deployment
```bash
# 1. Deploy system
python infrastructure/scripts/deployment.py

# 2. Start monitoring
python -c "from infrastructure.scripts.monitoring import SystemMonitor; m = SystemMonitor(); m.start()"

# 3. Create initial backup
python -c "from infrastructure.scripts.backup import BackupManager; b = BackupManager(); b.create_full_backup()"
```

### Regular Operations
```bash
# Run maintenance tasks
python infrastructure/scripts/maintenance.py

# Check system health
python -c "from infrastructure.scripts.monitoring import SystemMonitor; m = SystemMonitor(); m.start(); import time; time.sleep(60); print(m.get_system_summary())"

# Create backup
python -c "from infrastructure.scripts.backup import BackupManager; b = BackupManager(); b.create_full_backup()"
```

##   Monitoring Metrics

### System Metrics
- CPU usage percentage
- Memory usage percentage
- Disk usage percentage
- Network I/O (bytes sent/received)
- Active threads count
- Open files count

### Performance Metrics
- Order submission latency
- Trade execution latency
- Market data latency
- Risk calculation latency
- Average response times

### Business Metrics
- Orders per second
- Trades per second
- P&L tracking
- Fill rates
- Error rates

##   Alerting

### Alert Levels
- **INFO**: Informational messages
- **WARNING**: Warning conditions requiring attention
- **CRITICAL**: Critical conditions requiring immediate action

### Alert Channels
- **Email**: SMTP-based email notifications
- **Slack**: Webhook-based Slack messages
- **Logs**: All alerts logged to system logs

### Alert Thresholds
```python
thresholds = {
    'cpu_percent': 80.0,        # CPU > 80%
    'memory_percent': 85.0,     # Memory > 85%
    'disk_usage': 90.0,         # Disk > 90%
    'response_time': 1000,      # Latency > 1000ms
    'error_rate': 0.05          # Error rate > 5%
}
```

##   Backup Strategy

### Backup Types
- **Full Backup**: Complete system backup (weekly)
- **Incremental**: Changes since last backup (daily)
- **Differential**: Changes since last full backup

### Retention Policy
- Full backups: 4 weeks
- Incremental: 7 days
- Critical data: 90 days

### Recovery Procedures
```python
# List available backups
backups = manager.list_backups()

# Verify backup integrity
valid = manager.verify_backup(backup_id)

# Restore from backup
success = manager.restore_backup(backup_id)
```

##  ️ Troubleshooting

### Deployment Issues
```python
# Validate environment before deployment
deployer = SystemDeployer()
validations = deployer.validate_environment()
print(validations)

# Check deployment verification
verifications = deployer.verify_deployment()
print(verifications)
```

### Monitoring Issues
```python
# Check monitor status
monitor = SystemMonitor()
summary = monitor.get_system_summary()

# Export metrics for analysis
monitor.export_metrics('debug_metrics.json')
```

### Backup Issues
```python
# Verify backup integrity
manager = BackupManager()
is_valid = manager.verify_backup(backup_id)

# Check backup registry
backups = manager.list_backups()
for backup in backups:
    print(f"{backup.backup_id}: {backup.status}")
```

##   Best Practices

1. **Regular Backups**: Schedule daily backups with weekly full backups
2. **Monitoring**: Keep monitoring running 24/7 with appropriate thresholds
3. **Maintenance**: Run maintenance tasks during low-traffic periods
4. **Alerts**: Configure multiple alert channels for redundancy
5. **Testing**: Regularly test backup restoration procedures
6. **Documentation**: Maintain runbooks for common procedures
7. **Capacity Planning**: Monitor trends to predict resource needs

##   Security Considerations

- Store backup encryption keys securely
- Use secure connections for database operations
- Implement proper access controls for backup storage
- Audit log all administrative actions
- Rotate credentials regularly
- Enable backup encryption for sensitive data

##   Support

For infrastructure-related support:
- Check system logs in `logs/` directory
- Review monitoring dashboards
- Examine maintenance reports
- Verify backup integrity
- Contact DevOps team for critical issues

---

**Note**: These scripts are designed for production use. Always test in staging environment before deploying to production.
