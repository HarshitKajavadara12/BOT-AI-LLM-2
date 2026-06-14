"""
Configuration Management System for QUANTUM-FORGE
Advanced configuration management with validation, hot-reloading, and environment support.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading
import time
from pathlib import Path
import hashlib
import warnings
from copy import deepcopy
import importlib.util
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
warnings.filterwarnings('ignore')

@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    field_path: str
    rule_type: str  # 'type', 'range', 'choices', 'custom'
    rule_value: Any
    error_message: str = ""
    
    def validate(self, value: Any) -> bool:
        """Validate value against rule."""
        try:
            if self.rule_type == 'type':
                return isinstance(value, self.rule_value)
            elif self.rule_type == 'range':
                min_val, max_val = self.rule_value
                return min_val <= value <= max_val
            elif self.rule_type == 'choices':
                return value in self.rule_value
            elif self.rule_type == 'custom':
                return self.rule_value(value)
            return True
        except Exception:
            return False

@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    name: str
    version: str
    description: str
    validation_rules: List[ConfigValidationRule] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    
    def add_rule(self, field_path: str, rule_type: str, rule_value: Any, error_message: str = ""):
        """Add validation rule."""
        rule = ConfigValidationRule(field_path, rule_type, rule_value, error_message)
        self.validation_rules.append(rule)
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate configuration against schema."""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if not self._has_nested_field(config, field):
                errors.append(f"Required field missing: {field}")
        
        # Validate rules
        for rule in self.validation_rules:
            value = self._get_nested_value(config, rule.field_path)
            if value is not None and not rule.validate(value):
                error_msg = rule.error_message or f"Validation failed for {rule.field_path}"
                errors.append(error_msg)
        
        return len(errors) == 0, errors
    
    def _has_nested_field(self, config: Dict[str, Any], field_path: str) -> bool:
        """Check if nested field exists."""
        try:
            self._get_nested_value(config, field_path)
            return True
        except (KeyError, TypeError):
            return False
    
    def _get_nested_value(self, config: Dict[str, Any], field_path: str) -> Any:
        """Get nested configuration value."""
        keys = field_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value

class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes."""
    
    def __init__(self, config_manager):
        """Initialize watcher."""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            self.logger.info(f"[UPDATE] Configuration file changed: {event.src_path}")
            self.config_manager._reload_from_file(event.src_path)

class ConfigurationManager:
    """Advanced configuration management system."""
    
    def __init__(self, config_dir: str = "./config", enable_hot_reload: bool = True):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.enable_hot_reload = enable_hot_reload
        
        # Configuration storage
        self.configurations = {}
        self.schemas = {}
        self.environment = os.getenv('QUANTUM_FORGE_ENV', 'development')
        
        # Callbacks for configuration changes
        self.change_callbacks = {}
        
        # File watching
        self.observer = None
        self.file_watcher = None
        
        # Thread safety
        self.config_lock = threading.RLock()
        
        # Configuration history
        self.config_history = []
        self.max_history = 50
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize with default configuration
        self._load_default_configuration()
        
        # Setup file watching if enabled
        if self.enable_hot_reload:
            self._setup_file_watching()
        
        self.logger.info(f"[OK] Configuration Manager initialized (env: {self.environment})")
    
    def _load_default_configuration(self):
        """Load default QUANTUM-FORGE configuration."""
        default_config = {
            'system': {
                'name': 'QUANTUM-FORGE',
                'version': '1.0.0',
                'environment': self.environment,
                'debug_mode': self.environment == 'development',
                'log_level': 'INFO' if self.environment == 'production' else 'DEBUG',
                'max_workers': 8,
                'timezone': 'UTC'
            },
            'trading': {
                'default_currency': 'USD',
                'trading_hours': {
                    'start': '09:30:00',
                    'end': '16:00:00',
                    'timezone': 'America/New_York'
                },
                'position_limits': {
                    'max_position_size': 0.05,
                    'max_sector_exposure': 0.20,
                    'max_leverage': 3.0
                },
                'execution': {
                    'default_venue': 'SMART',
                    'order_timeout': 30,
                    'max_slippage_bps': 5.0
                }
            },
            'risk_management': {
                'var_confidence': 0.95,
                'var_horizon_days': 1,
                'stress_test_scenarios': 5,
                'correlation_threshold': 0.8,
                'volatility_lookback': 252,
                'max_drawdown_threshold': 0.20
            },
            'data': {
                'primary_source': 'reuters',
                'backup_source': 'bloomberg',
                'update_frequency_ms': 1000,
                'historical_retention_days': 365,
                'compression_enabled': True,
                'quality_threshold': 0.95
            },
            'strategies': {
                'rebalancing': {
                    'frequency': 'daily',
                    'min_rebalance_threshold': 0.05,
                    'transaction_cost_bps': 2.0
                },
                'signal_processing': {
                    'lookback_periods': [5, 10, 20, 50],
                    'smoothing_factor': 0.1,
                    'noise_filter_enabled': True
                }
            },
            'monitoring': {
                'alerts_enabled': True,
                'email_notifications': False,
                'slack_notifications': False,
                'performance_tracking': True,
                'health_check_interval': 60
            }
        }
        
        self.configurations['default'] = default_config
        
        # Create default schema
        self._create_default_schema()
    
    def _create_default_schema(self):
        """Create default configuration schema with validation rules."""
        schema = ConfigSchema(
            name="QUANTUM-FORGE Default Schema",
            version="1.0.0",
            description="Default configuration schema for QUANTUM-FORGE trading system"
        )
        
        # Add validation rules
        schema.add_rule('system.max_workers', 'range', (1, 32), 
                       "Max workers must be between 1 and 32")
        schema.add_rule('system.log_level', 'choices', ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       "Log level must be DEBUG, INFO, WARNING, or ERROR")
        
        schema.add_rule('trading.position_limits.max_position_size', 'range', (0.0, 1.0),
                       "Max position size must be between 0.0 and 1.0")
        schema.add_rule('trading.position_limits.max_leverage', 'range', (1.0, 10.0),
                       "Max leverage must be between 1.0 and 10.0")
        
        schema.add_rule('risk_management.var_confidence', 'range', (0.5, 0.999),
                       "VaR confidence must be between 0.5 and 0.999")
        schema.add_rule('risk_management.max_drawdown_threshold', 'range', (0.01, 1.0),
                       "Max drawdown threshold must be between 0.01 and 1.0")
        
        schema.add_rule('data.update_frequency_ms', 'range', (100, 10000),
                       "Update frequency must be between 100ms and 10000ms")
        schema.add_rule('data.quality_threshold', 'range', (0.0, 1.0),
                       "Quality threshold must be between 0.0 and 1.0")
        
        # Required fields
        schema.required_fields = [
            'system.name',
            'system.version',
            'trading.default_currency',
            'risk_management.var_confidence',
            'data.primary_source'
        ]
        
        self.schemas['default'] = schema
    
    def _setup_file_watching(self):
        """Setup file system watching for configuration changes."""
        try:
            self.file_watcher = ConfigFileWatcher(self)
            self.observer = Observer()
            self.observer.schedule(self.file_watcher, str(self.config_dir), recursive=True)
            self.observer.start()
            
            self.logger.info("[INIT] File watching enabled for configuration changes")
            
        except Exception as e:
            self.logger.warning(f"[WARN] Could not setup file watching: {e}")
    
    def get_config(self, config_name: str = 'default', section: str = None) -> Dict[str, Any]:
        """Get configuration or section."""
        with self.config_lock:
            if config_name not in self.configurations:
                self.logger.warning(f"⚠️ Configuration '{config_name}' not found")
                return {}
            
            config = self.configurations[config_name]
            
            if section:
                return config.get(section, {})
            
            return deepcopy(config)
    
    def set_config(self, config_name: str, config: Dict[str, Any], 
                  validate: bool = True, save_to_file: bool = True) -> bool:
        """Set configuration with optional validation."""
        try:
            with self.config_lock:
                # Validate if requested and schema exists
                if validate and config_name in self.schemas:
                    is_valid, errors = self.schemas[config_name].validate_config(config)
                    if not is_valid:
                        self.logger.error(f"[ERR] Configuration validation failed:")
                        for error in errors:
                            self.logger.error(f"   {error}")
                        return False
                
                # Store previous config for history
                old_config = self.configurations.get(config_name, {})
                
                # Update configuration
                self.configurations[config_name] = deepcopy(config)
                
                # Add to history
                self._add_to_history(config_name, old_config, config)
                
                # Save to file if requested
                if save_to_file:
                    self._save_to_file(config_name, config)
                
                # Notify callbacks
                self._notify_change_callbacks(config_name, config)
                
                self.logger.info(f"[OK] Configuration '{config_name}' updated successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"[ERR] Error setting configuration: {e}")
            return False
    
    def update_config_section(self, config_name: str, section: str, 
                            section_config: Dict[str, Any], validate: bool = True) -> bool:
        """Update specific configuration section."""
        try:
            with self.config_lock:
                if config_name not in self.configurations:
                    self.logger.error(f"❌ Configuration '{config_name}' not found")
                    return False
                
                # Get current config
                current_config = deepcopy(self.configurations[config_name])
                
                # Update section
                current_config[section] = section_config
                
                # Use set_config for validation and persistence
                return self.set_config(config_name, current_config, validate, save_to_file=True)
                
        except Exception as e:
            self.logger.error(f"❌ Error updating config section: {e}")
            return False
    
    def get_config_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """Get specific configuration value using dot notation."""
        try:
            with self.config_lock:
                if config_name not in self.configurations:
                    return default
                
                config = self.configurations[config_name]
                keys = key_path.split('.')
                value = config
                
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return default
                
                return value
                
        except Exception:
            return default
    
    def set_config_value(self, config_name: str, key_path: str, value: Any) -> bool:
        """Set specific configuration value using dot notation."""
        try:
            with self.config_lock:
                if config_name not in self.configurations:
                    self.logger.error(f"❌ Configuration '{config_name}' not found")
                    return False
                
                config = deepcopy(self.configurations[config_name])
                keys = key_path.split('.')
                
                # Navigate to parent of target key
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value
                current[keys[-1]] = value
                
                # Update the full config
                return self.set_config(config_name, config, validate=True)
                
        except Exception as e:
            self.logger.error(f"[ERR] Error setting config value: {e}")
            return False
    
    def load_from_file(self, file_path: str, config_name: str = None) -> bool:
        """Load configuration from file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"[ERR] Configuration file not found: {file_path}")
                return False
            
            # Determine config name
            if config_name is None:
                config_name = file_path.stem
            
            # Load based on file extension
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    config = json.load(f)
            else:
                self.logger.error(f"[ERR] Unsupported file format: {file_path.suffix}")
                return False
            
            # Set the configuration
            return self.set_config(config_name, config, validate=True, save_to_file=False)
            
        except Exception as e:
            self.logger.error(f"[ERR] Error loading configuration from file: {e}")
            return False
    
    def _save_to_file(self, config_name: str, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            # Create environment-specific filename
            filename = f"{config_name}_{self.environment}.yaml"
            file_path = self.config_dir / filename
            
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.debug(f"[SAVE] Configuration saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"[ERR] Error saving configuration to file: {e}")
    
    def _reload_from_file(self, file_path: str):
        """Reload configuration from file (used by file watcher)."""
        try:
            file_path = Path(file_path)
            
            # Determine config name from filename
            base_name = file_path.stem
            if f"_{self.environment}" in base_name:
                config_name = base_name.replace(f"_{self.environment}", "")
            else:
                config_name = base_name
            
            # Load and update
            if self.load_from_file(file_path, config_name):
                self.logger.info(f"🔄 Configuration '{config_name}' reloaded from file")
            
        except Exception as e:
            self.logger.error(f"❌ Error reloading configuration: {e}")
    
    def register_change_callback(self, config_name: str, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for configuration changes."""
        if config_name not in self.change_callbacks:
            self.change_callbacks[config_name] = []
        
        self.change_callbacks[config_name].append(callback)
        self.logger.info(f"📞 Registered change callback for '{config_name}'")
    
    def _notify_change_callbacks(self, config_name: str, new_config: Dict[str, Any]):
        """Notify registered callbacks of configuration changes."""
        if config_name in self.change_callbacks:
            for callback in self.change_callbacks[config_name]:
                try:
                    callback(new_config)
                except Exception as e:
                    self.logger.error(f"❌ Error in change callback: {e}")
    
    def _add_to_history(self, config_name: str, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Add configuration change to history."""
        history_entry = {
            'config_name': config_name,
            'timestamp': datetime.now(),
            'old_config_hash': self._hash_config(old_config),
            'new_config_hash': self._hash_config(new_config),
        }
        
        self.config_history.append(history_entry)
        
        # Maintain history size limit
        if len(self.config_history) > self.max_history:
            self.config_history = self.config_history[-self.max_history:]
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_configuration_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return deepcopy(self.config_history)
    
    def validate_configuration(self, config_name: str) -> tuple[bool, List[str]]:
        """Validate configuration against its schema."""
        if config_name not in self.configurations:
            return False, [f"Configuration '{config_name}' not found"]
        
        if config_name not in self.schemas:
            return True, []  # No schema means no validation
        
        return self.schemas[config_name].validate_config(self.configurations[config_name])
    
    def export_configuration(self, config_name: str, file_path: str, format: str = 'yaml') -> bool:
        """Export configuration to file."""
        try:
            if config_name not in self.configurations:
                self.logger.error(f"❌ Configuration '{config_name}' not found")
                return False
            
            config = self.configurations[config_name]
            file_path = Path(file_path)
            
            if format.lower() == 'yaml':
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                self.logger.error(f"[ERR] Unsupported export format: {format}")
                return False
            
            self.logger.info(f"[EXPORT] Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERR] Error exporting configuration: {e}")
            return False
    
    def list_configurations(self) -> List[str]:
        """List all available configurations."""
        return list(self.configurations.keys())
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'environment': self.environment,
            'config_directory': str(self.config_dir),
            'hot_reload_enabled': self.enable_hot_reload,
            'configurations_count': len(self.configurations),
            'schemas_count': len(self.schemas),
            'history_entries': len(self.config_history),
            'change_callbacks': {name: len(callbacks) for name, callbacks in self.change_callbacks.items()}
        }
    
    def shutdown(self):
        """Shutdown configuration manager."""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            self.logger.info("[STOP] Configuration Manager shut down")
            
        except Exception as e:
            self.logger.error(f"[ERR] Error during shutdown: {e}")

# Global configuration manager instance
_global_config_manager = None

def get_config_manager(config_dir: str = "./config", 
                      enable_hot_reload: bool = True) -> ConfigurationManager:
    """Get or create global configuration manager."""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigurationManager(config_dir, enable_hot_reload)
    
    return _global_config_manager

# Convenience functions
def get_config(section: str = None, config_name: str = 'default') -> Dict[str, Any]:
    """Get configuration section."""
    manager = get_config_manager()
    return manager.get_config(config_name, section)

def get_config_value(key_path: str, default: Any = None, config_name: str = 'default') -> Any:
    """Get specific configuration value."""
    manager = get_config_manager()
    return manager.get_config_value(config_name, key_path, default)

def set_config_value(key_path: str, value: Any, config_name: str = 'default') -> bool:
    """Set specific configuration value."""
    manager = get_config_manager()
    return manager.set_config_value(config_name, key_path, value)

# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration manager
    config_manager = ConfigurationManager("./test_config", enable_hot_reload=True)
    
    print("🔧 Testing QUANTUM-FORGE Configuration Management...")
    
    # Get default configuration
    default_config = config_manager.get_config('default')
    print(f"📋 Default config loaded with {len(default_config)} sections")
    
    # Test getting specific values
    log_level = config_manager.get_config_value('default', 'system.log_level')
    max_leverage = config_manager.get_config_value('default', 'trading.position_limits.max_leverage')
    print(f"📊 Log level: {log_level}, Max leverage: {max_leverage}")
    
    # Test validation
    is_valid, errors = config_manager.validate_configuration('default')
    print(f"✅ Configuration valid: {is_valid}")
    if errors:
        for error in errors:
            print(f"   ❌ {error}")
    
    # Test configuration change callback
    def config_change_callback(new_config):
        print(f"🔔 Configuration changed! New version has {len(new_config)} sections")
    
    config_manager.register_change_callback('default', config_change_callback)
    
    # Test setting a value
    success = config_manager.set_config_value('default', 'system.max_workers', 16)
    print(f"🔄 Setting max_workers to 16: {'Success' if success else 'Failed'}")
    
    # Test invalid value (should fail validation)
    success = config_manager.set_config_value('default', 'trading.position_limits.max_leverage', 15.0)
    print(f"🔄 Setting max_leverage to 15.0 (invalid): {'Success' if success else 'Failed'}")
    
    # Export configuration
    success = config_manager.export_configuration('default', './test_config_export.yaml')
    print(f"📤 Export configuration: {'Success' if success else 'Failed'}")
    
    # Show system info
    system_info = config_manager.get_system_info()
    print(f"\n🏥 System Info:")
    for key, value in system_info.items():
        print(f"   {key}: {value}")
    
    # Show configuration history
    history = config_manager.get_configuration_history()
    print(f"\n📚 Configuration History ({len(history)} entries):")
    for entry in history[-3:]:  # Show last 3 entries
        print(f"   {entry['timestamp']}: {entry['config_name']} changed")
    
    print("\n✅ Configuration management testing completed!")
    
    # Cleanup
    config_manager.shutdown()