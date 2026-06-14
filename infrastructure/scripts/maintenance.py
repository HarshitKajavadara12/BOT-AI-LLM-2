"""
System Maintenance Manager
Automated maintenance tasks for QUANTUM-FORGE
"""

import os
import sys
import shutil
import logging
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import schedule
import time
from dataclasses import dataclass
import hashlib


@dataclass
class MaintenanceTask:
    """Maintenance task definition."""
    name: str
    description: str
    schedule: str
    last_run: Optional[str]
    next_run: Optional[str]
    status: str
    
    
class MaintenanceManager:
    """Automated system maintenance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize maintenance manager.
        
        Args:
            config: Maintenance configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
        self.tasks = {}
        self.running = False
        self.maintenance_log = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default maintenance configuration."""
        return {
            'log_retention_days': 30,
            'data_retention_days': 90,
            'backup_retention_days': 7,
            'database_vacuum_schedule': 'daily',
            'cache_cleanup_schedule': 'hourly',
            'log_rotation_schedule': 'daily',
            'disk_cleanup_schedule': 'weekly',
            'health_check_schedule': 'hourly',
            'paths': {
                'logs': 'logs',
                'data': 'data',
                'backups': 'backups',
                'temp': 'temp'
            }
        }
    
    def initialize_tasks(self):
        """Initialize maintenance tasks."""
        # Database maintenance
        schedule.every().day.at("02:00").do(self.vacuum_database)
        
        # Cache cleanup
        schedule.every().hour.do(self.cleanup_cache)
        
        # Log rotation
        schedule.every().day.at("00:00").do(self.rotate_logs)
        
        # Disk cleanup
        schedule.every().sunday.at("03:00").do(self.cleanup_disk)
        
        # Health checks
        schedule.every().hour.do(self.run_health_checks)
        
        # Data archival
        schedule.every().day.at("04:00").do(self.archive_old_data)
        
        self.logger.info("Maintenance tasks initialized")
    
    def start(self):
        """Start maintenance scheduler."""
        if self.running:
            self.logger.warning("Maintenance manager already running")
            return
        
        self.running = True
        self.initialize_tasks()
        
        self.logger.info("Maintenance manager started")
        
        # Run scheduler in loop
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop maintenance scheduler."""
        self.running = False
        schedule.clear()
        self.logger.info("Maintenance manager stopped")
    
    def vacuum_database(self):
        """Vacuum and optimize database."""
        try:
            self.logger.info("Starting database vacuum...")
            
            from sqlalchemy import create_engine
            
            db_config = self.config.get('database', {})
            engine = create_engine(
                f"postgresql://{db_config.get('host', 'localhost')}:"
                f"{db_config.get('port', 5432)}/{db_config.get('name', 'quantum_forge')}"
            )
            
            with engine.connect() as conn:
                # Vacuum and analyze
                conn.execute("VACUUM ANALYZE")
            
            self._log_task_completion('vacuum_database', 'success')
            self.logger.info("Database vacuum completed")
            
        except Exception as e:
            self._log_task_completion('vacuum_database', 'failed', str(e))
            self.logger.error(f"Database vacuum failed: {e}")
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        try:
            self.logger.info("Starting cache cleanup...")
            
            import redis
            
            redis_config = self.config.get('redis', {})
            r = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0)
            )
            
            # Get keys with expiration
            total_keys = 0
            expired_keys = 0
            
            for key in r.scan_iter():
                total_keys += 1
                ttl = r.ttl(key)
                if ttl == -1:  # No expiration set
                    # Set default expiration for orphaned keys
                    r.expire(key, 86400)  # 24 hours
            
            self._log_task_completion('cleanup_cache', 'success', 
                                    f"Processed {total_keys} keys")
            self.logger.info(f"Cache cleanup completed: {total_keys} keys processed")
            
        except Exception as e:
            self._log_task_completion('cleanup_cache', 'failed', str(e))
            self.logger.error(f"Cache cleanup failed: {e}")
    
    def rotate_logs(self):
        """Rotate and compress log files."""
        try:
            self.logger.info("Starting log rotation...")
            
            logs_path = Path(self.config['paths']['logs'])
            if not logs_path.exists():
                return
            
            rotated_count = 0
            compressed_count = 0
            
            # Find log files
            for log_file in logs_path.glob('*.log'):
                # Get file modification time
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                age_days = (datetime.now() - mtime).days
                
                # Rotate if older than 1 day
                if age_days >= 1:
                    # Create compressed archive
                    archive_name = f"{log_file.stem}_{mtime.strftime('%Y%m%d')}.log.gz"
                    archive_path = logs_path / 'archive' / archive_name
                    
                    archive_path.parent.mkdir(exist_ok=True)
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(archive_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original
                    log_file.unlink()
                    
                    rotated_count += 1
                    compressed_count += 1
            
            # Clean up old archives
            retention_days = self.config['log_retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            archive_path = logs_path / 'archive'
            if archive_path.exists():
                for archive_file in archive_path.glob('*.log.gz'):
                    mtime = datetime.fromtimestamp(archive_file.stat().st_mtime)
                    if mtime < cutoff_date:
                        archive_file.unlink()
            
            self._log_task_completion('rotate_logs', 'success',
                                    f"Rotated {rotated_count} files, compressed {compressed_count}")
            self.logger.info(f"Log rotation completed: {rotated_count} files processed")
            
        except Exception as e:
            self._log_task_completion('rotate_logs', 'failed', str(e))
            self.logger.error(f"Log rotation failed: {e}")
    
    def cleanup_disk(self):
        """Clean up temporary files and old data."""
        try:
            self.logger.info("Starting disk cleanup...")
            
            cleaned_size = 0
            files_removed = 0
            
            # Clean temp directory
            temp_path = Path(self.config['paths']['temp'])
            if temp_path.exists():
                for temp_file in temp_path.glob('*'):
                    try:
                        if temp_file.is_file():
                            size = temp_file.stat().st_size
                            temp_file.unlink()
                            cleaned_size += size
                            files_removed += 1
                        elif temp_file.is_dir():
                            size = sum(f.stat().st_size for f in temp_file.rglob('*') if f.is_file())
                            shutil.rmtree(temp_file)
                            cleaned_size += size
                            files_removed += 1
                    except Exception as e:
                        self.logger.warning(f"Could not remove {temp_file}: {e}")
            
            # Clean old data files
            data_path = Path(self.config['paths']['data'])
            if data_path.exists():
                retention_days = self.config['data_retention_days']
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                for data_file in data_path.rglob('*'):
                    if data_file.is_file():
                        mtime = datetime.fromtimestamp(data_file.stat().st_mtime)
                        if mtime < cutoff_date:
                            try:
                                size = data_file.stat().st_size
                                data_file.unlink()
                                cleaned_size += size
                                files_removed += 1
                            except Exception as e:
                                self.logger.warning(f"Could not remove {data_file}: {e}")
            
            cleaned_mb = cleaned_size / (1024 * 1024)
            self._log_task_completion('cleanup_disk', 'success',
                                    f"Removed {files_removed} files, freed {cleaned_mb:.2f} MB")
            self.logger.info(f"Disk cleanup completed: {cleaned_mb:.2f} MB freed")
            
        except Exception as e:
            self._log_task_completion('cleanup_disk', 'failed', str(e))
            self.logger.error(f"Disk cleanup failed: {e}")
    
    def run_health_checks(self):
        """Run system health checks."""
        try:
            self.logger.info("Starting health checks...")
            
            health_status = {}
            
            # Check disk space
            import psutil
            disk = psutil.disk_usage('/')
            health_status['disk_available_percent'] = 100 - disk.percent
            
            # Check memory
            memory = psutil.virtual_memory()
            health_status['memory_available_percent'] = 100 - memory.percent
            
            # Check database connectivity
            try:
                from sqlalchemy import create_engine
                db_config = self.config.get('database', {})
                engine = create_engine(
                    f"postgresql://{db_config.get('host', 'localhost')}:"
                    f"{db_config.get('port', 5432)}/{db_config.get('name', 'quantum_forge')}"
                )
                with engine.connect():
                    health_status['database_connected'] = True
            except:
                health_status['database_connected'] = False
            
            # Check Redis connectivity
            try:
                import redis
                redis_config = self.config.get('redis', {})
                r = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0)
                )
                r.ping()
                health_status['redis_connected'] = True
            except:
                health_status['redis_connected'] = False
            
            # Check critical directories
            for name, path in self.config['paths'].items():
                health_status[f'path_{name}_exists'] = os.path.exists(path)
            
            # Determine overall health
            all_healthy = all([
                health_status['disk_available_percent'] > 10,
                health_status['memory_available_percent'] > 10,
                health_status.get('database_connected', False),
                health_status.get('redis_connected', False)
            ])
            
            status = 'healthy' if all_healthy else 'unhealthy'
            
            self._log_task_completion('health_checks', status, 
                                    json.dumps(health_status))
            self.logger.info(f"Health checks completed: {status}")
            
            return health_status
            
        except Exception as e:
            self._log_task_completion('health_checks', 'failed', str(e))
            self.logger.error(f"Health checks failed: {e}")
            return {}
    
    def archive_old_data(self):
        """Archive old data to compressed storage."""
        try:
            self.logger.info("Starting data archival...")
            
            archived_count = 0
            archived_size = 0
            
            data_path = Path(self.config['paths']['data'])
            archive_path = data_path / 'archive'
            archive_path.mkdir(exist_ok=True)
            
            # Archive data older than retention period
            retention_days = self.config['data_retention_days']
            archive_date = datetime.now() - timedelta(days=retention_days)
            
            if data_path.exists():
                for data_file in data_path.glob('*.csv'):
                    mtime = datetime.fromtimestamp(data_file.stat().st_mtime)
                    
                    if mtime < archive_date:
                        # Create archive
                        archive_name = f"{data_file.stem}_{mtime.strftime('%Y%m%d')}.csv.gz"
                        archive_file = archive_path / archive_name
                        
                        with open(data_file, 'rb') as f_in:
                            with gzip.open(archive_file, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        size = data_file.stat().st_size
                        data_file.unlink()
                        
                        archived_count += 1
                        archived_size += size
            
            archived_mb = archived_size / (1024 * 1024)
            self._log_task_completion('archive_data', 'success',
                                    f"Archived {archived_count} files, {archived_mb:.2f} MB")
            self.logger.info(f"Data archival completed: {archived_count} files archived")
            
        except Exception as e:
            self._log_task_completion('archive_data', 'failed', str(e))
            self.logger.error(f"Data archival failed: {e}")
    
    def optimize_database_indices(self):
        """Optimize database indices."""
        try:
            self.logger.info("Starting database index optimization...")
            
            from sqlalchemy import create_engine, text
            
            db_config = self.config.get('database', {})
            engine = create_engine(
                f"postgresql://{db_config.get('host', 'localhost')}:"
                f"{db_config.get('port', 5432)}/{db_config.get('name', 'quantum_forge')}"
            )
            
            with engine.connect() as conn:
                # Reindex all tables
                result = conn.execute(text("REINDEX DATABASE quantum_forge"))
            
            self._log_task_completion('optimize_indices', 'success')
            self.logger.info("Database index optimization completed")
            
        except Exception as e:
            self._log_task_completion('optimize_indices', 'failed', str(e))
            self.logger.error(f"Database index optimization failed: {e}")
    
    def verify_data_integrity(self):
        """Verify data integrity with checksums."""
        try:
            self.logger.info("Starting data integrity verification...")
            
            data_path = Path(self.config['paths']['data'])
            checksum_file = data_path / 'checksums.json'
            
            # Load existing checksums
            existing_checksums = {}
            if checksum_file.exists():
                with open(checksum_file, 'r') as f:
                    existing_checksums = json.load(f)
            
            # Calculate current checksums
            current_checksums = {}
            corrupted_files = []
            
            if data_path.exists():
                for data_file in data_path.rglob('*'):
                    if data_file.is_file() and data_file != checksum_file:
                        with open(data_file, 'rb') as f:
                            file_hash = hashlib.sha256(f.read()).hexdigest()
                            current_checksums[str(data_file)] = file_hash
                            
                            # Check against existing
                            existing_hash = existing_checksums.get(str(data_file))
                            if existing_hash and existing_hash != file_hash:
                                corrupted_files.append(str(data_file))
            
            # Save current checksums
            with open(checksum_file, 'w') as f:
                json.dump(current_checksums, f, indent=2)
            
            status = 'success' if not corrupted_files else 'warning'
            message = f"Verified {len(current_checksums)} files"
            if corrupted_files:
                message += f", {len(corrupted_files)} corrupted"
            
            self._log_task_completion('verify_integrity', status, message)
            self.logger.info(f"Data integrity verification completed: {message}")
            
            return corrupted_files
            
        except Exception as e:
            self._log_task_completion('verify_integrity', 'failed', str(e))
            self.logger.error(f"Data integrity verification failed: {e}")
            return []
    
    def _log_task_completion(self, task_name: str, status: str, details: str = ''):
        """Log maintenance task completion."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'task': task_name,
            'status': status,
            'details': details
        }
        self.maintenance_log.append(log_entry)
        
        # Keep only recent logs
        if len(self.maintenance_log) > 1000:
            self.maintenance_log = self.maintenance_log[-1000:]
    
    def get_maintenance_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get maintenance task history."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            log for log in self.maintenance_log
            if datetime.fromisoformat(log['timestamp']) > cutoff
        ]
    
    def generate_maintenance_report(self, output_path: str = 'maintenance_report.json'):
        """Generate maintenance report."""
        try:
            history = self.get_maintenance_history(hours=168)  # Last week
            
            # Calculate statistics
            task_stats = {}
            for log in history:
                task = log['task']
                if task not in task_stats:
                    task_stats[task] = {'success': 0, 'failed': 0, 'warning': 0}
                task_stats[task][log['status']] = task_stats[task].get(log['status'], 0) + 1
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'period_hours': 168,
                'total_tasks': len(history),
                'task_statistics': task_stats,
                'recent_tasks': history[-20:]  # Last 20 tasks
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Maintenance report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate maintenance report: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run maintenance tasks
    manager = MaintenanceManager()
    
    # Run immediate tasks
    manager.cleanup_cache()
    manager.cleanup_disk()
    manager.run_health_checks()
    
    # Generate report
    manager.generate_maintenance_report()
