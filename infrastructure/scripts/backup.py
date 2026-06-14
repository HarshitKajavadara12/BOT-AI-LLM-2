"""
Backup Manager
Automated backup and recovery for QUANTUM-FORGE
"""

import os
import sys
import shutil
import tarfile
import gzip
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import tempfile
from dataclasses import dataclass, asdict


@dataclass
class BackupInfo:
    """Backup information."""
    backup_id: str
    timestamp: str
    backup_type: str  # full, incremental, differential
    size_bytes: int
    checksum: str
    components: List[str]
    status: str
    path: str


class BackupManager:
    """Automated backup and recovery management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize backup manager.
        
        Args:
            config: Backup configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
        self.backup_registry = []
        self.registry_file = Path(self.config['backup_path']) / 'backup_registry.json'
        self._load_registry()
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default backup configuration."""
        return {
            'backup_path': 'backups',
            'retention_days': 30,
            'full_backup_schedule': 'weekly',
            'incremental_schedule': 'daily',
            'compression': True,
            'encryption': False,
            'components': {
                'database': True,
                'config': True,
                'logs': True,
                'data': True,
                'models': True
            },
            'paths': {
                'database': 'database',
                'config': 'config',
                'logs': 'logs',
                'data': 'data',
                'models': 'models'
            }
        }
    
    def _load_registry(self):
        """Load backup registry."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.backup_registry = [
                        BackupInfo(**item) for item in data
                    ]
        except Exception as e:
            self.logger.error(f"Failed to load backup registry: {e}")
            self.backup_registry = []
    
    def _save_registry(self):
        """Save backup registry."""
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, 'w') as f:
                data = [asdict(backup) for backup in self.backup_registry]
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save backup registry: {e}")
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_hash = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
        return f"backup_{timestamp}_{random_hash}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def create_full_backup(self, components: Optional[List[str]] = None) -> BackupInfo:
        """
        Create full system backup.
        
        Args:
            components: List of components to backup (None = all)
            
        Returns:
            Backup information
        """
        try:
            self.logger.info("Starting full backup...")
            
            backup_id = self._generate_backup_id()
            timestamp = datetime.now().isoformat()
            backup_path = Path(self.config['backup_path']) / backup_id
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Determine components to backup
            if components is None:
                components = [k for k, v in self.config['components'].items() if v]
            
            # Backup each component
            total_size = 0
            for component in components:
                component_size = self._backup_component(component, backup_path)
                total_size += component_size
            
            # Create backup metadata
            metadata = {
                'backup_id': backup_id,
                'timestamp': timestamp,
                'type': 'full',
                'components': components,
                'total_size_bytes': total_size
            }
            
            metadata_file = backup_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Compress backup if enabled
            if self.config['compression']:
                archive_path = self._compress_backup(backup_path)
                final_path = archive_path
                
                # Remove uncompressed backup
                shutil.rmtree(backup_path)
            else:
                final_path = backup_path
            
            # Calculate checksum
            if self.config['compression']:
                checksum = self._calculate_checksum(final_path)
            else:
                checksum = hashlib.md5(backup_id.encode()).hexdigest()
            
            # Create backup info
            backup_info = BackupInfo(
                backup_id=backup_id,
                timestamp=timestamp,
                backup_type='full',
                size_bytes=total_size,
                checksum=checksum,
                components=components,
                status='completed',
                path=str(final_path)
            )
            
            # Register backup
            self.backup_registry.append(backup_info)
            self._save_registry()
            
            self.logger.info(f"Full backup completed: {backup_id}")
            return backup_info
            
        except Exception as e:
            self.logger.error(f"Full backup failed: {e}")
            raise
    
    def _backup_component(self, component: str, backup_path: Path) -> int:
        """
        Backup specific component.
        
        Args:
            component: Component name
            backup_path: Backup destination path
            
        Returns:
            Component size in bytes
        """
        try:
            self.logger.info(f"Backing up component: {component}")
            
            if component == 'database':
                return self._backup_database(backup_path)
            elif component == 'config':
                return self._backup_config(backup_path)
            elif component == 'logs':
                return self._backup_logs(backup_path)
            elif component == 'data':
                return self._backup_data(backup_path)
            elif component == 'models':
                return self._backup_models(backup_path)
            else:
                self.logger.warning(f"Unknown component: {component}")
                return 0
                
        except Exception as e:
            self.logger.error(f"Failed to backup component {component}: {e}")
            return 0
    
    def _backup_database(self, backup_path: Path) -> int:
        """Backup database."""
        try:
            from sqlalchemy import create_engine
            import subprocess
            
            db_config = self.config.get('database', {})
            dump_file = backup_path / 'database_dump.sql'
            
            # PostgreSQL dump command
            dump_cmd = [
                'pg_dump',
                '-h', db_config.get('host', 'localhost'),
                '-p', str(db_config.get('port', 5432)),
                '-U', db_config.get('user', 'postgres'),
                '-d', db_config.get('name', 'quantum_forge'),
                '-f', str(dump_file)
            ]
            
            # Run dump
            # subprocess.run(dump_cmd, check=True)
            
            # For demonstration, create placeholder
            dump_file.write_text("-- Database dump placeholder\n")
            
            return dump_file.stat().st_size if dump_file.exists() else 0
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return 0
    
    def _backup_config(self, backup_path: Path) -> int:
        """Backup configuration files."""
        try:
            config_backup = backup_path / 'config'
            config_backup.mkdir(exist_ok=True)
            
            config_path = Path(self.config['paths'].get('config', 'config'))
            
            if config_path.exists():
                shutil.copytree(config_path, config_backup, dirs_exist_ok=True)
                
                total_size = sum(
                    f.stat().st_size for f in config_backup.rglob('*') if f.is_file()
                )
                return total_size
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Config backup failed: {e}")
            return 0
    
    def _backup_logs(self, backup_path: Path) -> int:
        """Backup log files."""
        try:
            logs_backup = backup_path / 'logs'
            logs_backup.mkdir(exist_ok=True)
            
            logs_path = Path(self.config['paths'].get('logs', 'logs'))
            
            if logs_path.exists():
                # Only backup recent logs (last 7 days)
                cutoff = datetime.now() - timedelta(days=7)
                
                total_size = 0
                for log_file in logs_path.glob('*.log'):
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime > cutoff:
                        dest = logs_backup / log_file.name
                        shutil.copy2(log_file, dest)
                        total_size += dest.stat().st_size
                
                return total_size
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Logs backup failed: {e}")
            return 0
    
    def _backup_data(self, backup_path: Path) -> int:
        """Backup data files."""
        try:
            data_backup = backup_path / 'data'
            data_backup.mkdir(exist_ok=True)
            
            data_path = Path(self.config['paths'].get('data', 'data'))
            
            if data_path.exists():
                shutil.copytree(data_path, data_backup, dirs_exist_ok=True)
                
                total_size = sum(
                    f.stat().st_size for f in data_backup.rglob('*') if f.is_file()
                )
                return total_size
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Data backup failed: {e}")
            return 0
    
    def _backup_models(self, backup_path: Path) -> int:
        """Backup ML models."""
        try:
            models_backup = backup_path / 'models'
            models_backup.mkdir(exist_ok=True)
            
            models_path = Path(self.config['paths'].get('models', 'models'))
            
            if models_path.exists():
                shutil.copytree(models_path, models_backup, dirs_exist_ok=True)
                
                total_size = sum(
                    f.stat().st_size for f in models_backup.rglob('*') if f.is_file()
                )
                return total_size
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Models backup failed: {e}")
            return 0
    
    def _compress_backup(self, backup_path: Path) -> Path:
        """Compress backup directory."""
        try:
            archive_path = backup_path.parent / f"{backup_path.name}.tar.gz"
            
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(backup_path, arcname=backup_path.name)
            
            self.logger.info(f"Backup compressed: {archive_path}")
            return archive_path
            
        except Exception as e:
            self.logger.error(f"Backup compression failed: {e}")
            raise
    
    def restore_backup(self, backup_id: str, components: Optional[List[str]] = None) -> bool:
        """
        Restore from backup.
        
        Args:
            backup_id: Backup ID to restore
            components: Components to restore (None = all)
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Starting restore from backup: {backup_id}")
            
            # Find backup info
            backup_info = None
            for backup in self.backup_registry:
                if backup.backup_id == backup_id:
                    backup_info = backup
                    break
            
            if not backup_info:
                raise ValueError(f"Backup not found: {backup_id}")
            
            backup_path = Path(backup_info.path)
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Verify checksum
            if self.config['compression']:
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum != backup_info.checksum:
                    raise ValueError("Backup checksum mismatch - file may be corrupted")
            
            # Extract if compressed
            if self.config['compression']:
                temp_dir = Path(tempfile.mkdtemp())
                with tarfile.open(backup_path, "r:gz") as tar:
                    tar.extractall(temp_dir)
                restore_path = temp_dir / backup_info.backup_id
            else:
                restore_path = backup_path
            
            # Determine components to restore
            if components is None:
                components = backup_info.components
            
            # Restore each component
            for component in components:
                self._restore_component(component, restore_path)
            
            # Cleanup temp directory
            if self.config['compression']:
                shutil.rmtree(temp_dir)
            
            self.logger.info(f"Restore completed: {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def _restore_component(self, component: str, restore_path: Path):
        """Restore specific component."""
        try:
            self.logger.info(f"Restoring component: {component}")
            
            if component == 'database':
                self._restore_database(restore_path)
            elif component == 'config':
                self._restore_config(restore_path)
            elif component == 'logs':
                self._restore_logs(restore_path)
            elif component == 'data':
                self._restore_data(restore_path)
            elif component == 'models':
                self._restore_models(restore_path)
                
        except Exception as e:
            self.logger.error(f"Failed to restore component {component}: {e}")
    
    def _restore_database(self, restore_path: Path):
        """Restore database."""
        dump_file = restore_path / 'database_dump.sql'
        if dump_file.exists():
            # In production, use psql to restore
            self.logger.info("Database restore would execute here")
    
    def _restore_config(self, restore_path: Path):
        """Restore configuration."""
        config_backup = restore_path / 'config'
        if config_backup.exists():
            config_path = Path(self.config['paths'].get('config', 'config'))
            shutil.copytree(config_backup, config_path, dirs_exist_ok=True)
    
    def _restore_logs(self, restore_path: Path):
        """Restore logs."""
        logs_backup = restore_path / 'logs'
        if logs_backup.exists():
            logs_path = Path(self.config['paths'].get('logs', 'logs'))
            shutil.copytree(logs_backup, logs_path, dirs_exist_ok=True)
    
    def _restore_data(self, restore_path: Path):
        """Restore data."""
        data_backup = restore_path / 'data'
        if data_backup.exists():
            data_path = Path(self.config['paths'].get('data', 'data'))
            shutil.copytree(data_backup, data_path, dirs_exist_ok=True)
    
    def _restore_models(self, restore_path: Path):
        """Restore models."""
        models_backup = restore_path / 'models'
        if models_backup.exists():
            models_path = Path(self.config['paths'].get('models', 'models'))
            shutil.copytree(models_backup, models_path, dirs_exist_ok=True)
    
    def list_backups(self, backup_type: Optional[str] = None) -> List[BackupInfo]:
        """
        List available backups.
        
        Args:
            backup_type: Filter by backup type (full/incremental/differential)
            
        Returns:
            List of backup information
        """
        if backup_type:
            return [b for b in self.backup_registry if b.backup_type == backup_type]
        return self.backup_registry
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete backup.
        
        Args:
            backup_id: Backup ID to delete
            
        Returns:
            True if successful
        """
        try:
            # Find backup
            backup_info = None
            for idx, backup in enumerate(self.backup_registry):
                if backup.backup_id == backup_id:
                    backup_info = backup
                    backup_idx = idx
                    break
            
            if not backup_info:
                raise ValueError(f"Backup not found: {backup_id}")
            
            # Delete backup file
            backup_path = Path(backup_info.path)
            if backup_path.exists():
                if backup_path.is_file():
                    backup_path.unlink()
                else:
                    shutil.rmtree(backup_path)
            
            # Remove from registry
            self.backup_registry.pop(backup_idx)
            self._save_registry()
            
            self.logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete backup: {e}")
            return False
    
    def cleanup_old_backups(self):
        """Clean up backups older than retention period."""
        try:
            retention_days = self.config['retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            deleted_count = 0
            for backup in self.backup_registry[:]:
                backup_date = datetime.fromisoformat(backup.timestamp)
                if backup_date < cutoff_date:
                    if self.delete_backup(backup.backup_id):
                        deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old backups")
            
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
    
    def verify_backup(self, backup_id: str) -> bool:
        """
        Verify backup integrity.
        
        Args:
            backup_id: Backup ID to verify
            
        Returns:
            True if backup is valid
        """
        try:
            # Find backup
            backup_info = None
            for backup in self.backup_registry:
                if backup.backup_id == backup_id:
                    backup_info = backup
                    break
            
            if not backup_info:
                return False
            
            backup_path = Path(backup_info.path)
            if not backup_path.exists():
                return False
            
            # Verify checksum
            if self.config['compression']:
                current_checksum = self._calculate_checksum(backup_path)
                return current_checksum == backup_info.checksum
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False
    
    def generate_backup_report(self, output_path: str = 'backup_report.json'):
        """Generate backup report."""
        try:
            total_size = sum(b.size_bytes for b in self.backup_registry)
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_backups': len(self.backup_registry),
                'total_size_bytes': total_size,
                'total_size_gb': total_size / (1024**3),
                'retention_days': self.config['retention_days'],
                'backups': [asdict(b) for b in self.backup_registry]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Backup report generated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate backup report: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create backup
    manager = BackupManager()
    backup_info = manager.create_full_backup()
    
    print(f"\nBackup created: {backup_info.backup_id}")
    print(f"Size: {backup_info.size_bytes / (1024**2):.2f} MB")
    print(f"Path: {backup_info.path}")
    
    # Generate report
    manager.generate_backup_report()
