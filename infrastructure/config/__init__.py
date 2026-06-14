"""
Configuration System Initialization for QUANTUM-FORGE
Environment management, parameter validation, and hot-reloading capabilities.
"""

from .config_manager import (
    ConfigurationManager, ConfigSchema, ConfigValidationRule,
    get_config_manager, get_config, get_config_value, set_config_value
)

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class QuantumForgeConfig:
    """Centralized QUANTUM-FORGE configuration access."""
    
    def __init__(self):
        """Initialize QUANTUM-FORGE configuration system."""
        self.manager = get_config_manager()
        self.logger = logging.getLogger(__name__)
        
        # Load environment-specific configurations
        self._load_environment_configs()
        
        # Setup configuration profiles
        self._setup_configuration_profiles()
        
        self.logger.info("  QUANTUM-FORGE Configuration System initialized")
    
    def _load_environment_configs(self):
        """Load environment-specific configurations."""
        env = os.getenv('QUANTUM_FORGE_ENV', 'development')
        
        # Define environment-specific overrides
        env_configs = {
            'development': {
                'system.debug_mode': True,
                'system.log_level': 'DEBUG',
                'monitoring.alerts_enabled': False,
                'data.update_frequency_ms': 5000,  # Slower updates for dev
                'trading.position_limits.max_leverage': 2.0  # Lower leverage for dev
            },
            'testing': {
                'system.debug_mode': True,
                'system.log_level': 'INFO',
                'monitoring.alerts_enabled': False,
                'data.update_frequency_ms': 1000,
                'trading.execution.order_timeout': 10  # Faster timeout for tests
            },
            'staging': {
                'system.debug_mode': False,
                'system.log_level': 'INFO',
                'monitoring.alerts_enabled': True,
                'data.update_frequency_ms': 1000,
                'trading.position_limits.max_leverage': 2.5
            },
            'production': {
                'system.debug_mode': False,
                'system.log_level': 'WARNING',
                'monitoring.alerts_enabled': True,
                'monitoring.email_notifications': True,
                'data.update_frequency_ms': 500,  # Fastest updates for prod
                'trading.position_limits.max_leverage': 3.0
            }
        }
        
        # Apply environment-specific settings
        if env in env_configs:
            for key, value in env_configs[env].items():
                self.manager.set_config_value('default', key, value)
            
            self.logger.info(f"  Applied {env} environment configuration")
    
    def _setup_configuration_profiles(self):
        """Setup different configuration profiles for various use cases."""
        
        # Conservative trading profile
        conservative_config = self.manager.get_config('default')
        conservative_config.update({
            'trading': {
                **conservative_config['trading'],
                'position_limits': {
                    'max_position_size': 0.02,  # 2% max position
                    'max_sector_exposure': 0.10,  # 10% sector exposure
                    'max_leverage': 1.5  # Low leverage
                }
            },
            'risk_management': {
                **conservative_config['risk_management'],
                'var_confidence': 0.99,  # Higher confidence
                'max_drawdown_threshold': 0.10  # Lower drawdown tolerance
            }
        })
        
        self.manager.set_config('conservative', conservative_config, validate=True)
        
        # Aggressive trading profile
        aggressive_config = self.manager.get_config('default')
        aggressive_config.update({
            'trading': {
                **aggressive_config['trading'],
                'position_limits': {
                    'max_position_size': 0.10,  # 10% max position
                    'max_sector_exposure': 0.30,  # 30% sector exposure
                    'max_leverage': 5.0  # Higher leverage
                }
            },
            'risk_management': {
                **aggressive_config['risk_management'],
                'var_confidence': 0.90,  # Lower confidence (more aggressive)
                'max_drawdown_threshold': 0.30  # Higher drawdown tolerance
            }
        })
        
        self.manager.set_config('aggressive', aggressive_config, validate=True)
        
        # Research profile (for backtesting and analysis)
        research_config = self.manager.get_config('default')
        research_config.update({
            'data': {
                **research_config['data'],
                'historical_retention_days': 1825,  # 5 years of data
                'update_frequency_ms': 10000,  # Slower updates for research
            },
            'strategies': {
                **research_config['strategies'],
                'signal_processing': {
                    'lookback_periods': [1, 5, 10, 20, 50, 100, 200],  # More periods
                    'smoothing_factor': 0.05,  # Less smoothing
                    'noise_filter_enabled': False  # No filtering for research
                }
            }
        })
        
        self.manager.set_config('research', research_config, validate=True)
        
        self.logger.info("  Configuration profiles created: conservative, aggressive, research")
    
    # Convenience methods for common configuration access
    def get_trading_config(self, profile: str = 'default') -> Dict[str, Any]:
        """Get trading configuration."""
        return self.manager.get_config(profile, 'trading')
    
    def get_risk_config(self, profile: str = 'default') -> Dict[str, Any]:
        """Get risk management configuration."""
        return self.manager.get_config(profile, 'risk_management')
    
    def get_data_config(self, profile: str = 'default') -> Dict[str, Any]:
        """Get data configuration."""
        return self.manager.get_config(profile, 'data')
    
    def get_system_config(self, profile: str = 'default') -> Dict[str, Any]:
        """Get system configuration."""
        return self.manager.get_config(profile, 'system')
    
    def get_monitoring_config(self, profile: str = 'default') -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.manager.get_config(profile, 'monitoring')
    
    def is_debug_mode(self, profile: str = 'default') -> bool:
        """Check if debug mode is enabled."""
        return self.manager.get_config_value(profile, 'system.debug_mode', False)
    
    def get_log_level(self, profile: str = 'default') -> str:
        """Get log level."""
        return self.manager.get_config_value(profile, 'system.log_level', 'INFO')
    
    def get_max_position_size(self, profile: str = 'default') -> float:
        """Get maximum position size."""
        return self.manager.get_config_value(profile, 'trading.position_limits.max_position_size', 0.05)
    
    def get_max_leverage(self, profile: str = 'default') -> float:
        """Get maximum leverage."""
        return self.manager.get_config_value(profile, 'trading.position_limits.max_leverage', 3.0)
    
    def get_var_confidence(self, profile: str = 'default') -> float:
        """Get VaR confidence level."""
        return self.manager.get_config_value(profile, 'risk_management.var_confidence', 0.95)
    
    def get_update_frequency(self, profile: str = 'default') -> int:
        """Get data update frequency in milliseconds."""
        return self.manager.get_config_value(profile, 'data.update_frequency_ms', 1000)
    
    def switch_profile(self, profile: str) -> bool:
        """Switch to a different configuration profile."""
        if profile in self.manager.list_configurations():
            os.environ['QUANTUM_FORGE_PROFILE'] = profile
            self.logger.info(f"  Switched to configuration profile: {profile}")
            return True
        else:
            self.logger.error(f"  Configuration profile not found: {profile}")
            return False
    
    def get_current_profile(self) -> str:
        """Get current configuration profile."""
        return os.getenv('QUANTUM_FORGE_PROFILE', 'default')
    
    def list_profiles(self) -> list:
        """List available configuration profiles."""
        return self.manager.list_configurations()
    
    def validate_current_config(self, profile: str = None) -> tuple[bool, list]:
        """Validate current configuration."""
        profile = profile or self.get_current_profile()
        return self.manager.validate_configuration(profile)
    
    def export_config(self, profile: str, file_path: str, format: str = 'yaml') -> bool:
        """Export configuration to file."""
        return self.manager.export_configuration(profile, file_path, format)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration system summary."""
        return {
            'current_profile': self.get_current_profile(),
            'available_profiles': self.list_profiles(),
            'environment': os.getenv('QUANTUM_FORGE_ENV', 'development'),
            'system_info': self.manager.get_system_info(),
            'key_settings': {
                'debug_mode': self.is_debug_mode(),
                'log_level': self.get_log_level(),
                'max_position_size': self.get_max_position_size(),
                'max_leverage': self.get_max_leverage(),
                'update_frequency_ms': self.get_update_frequency()
            }
        }

# Global configuration instance
_global_quantum_config = None

def get_quantum_config() -> QuantumForgeConfig:
    """Get or create global QUANTUM-FORGE configuration instance."""
    global _global_quantum_config
    
    if _global_quantum_config is None:
        _global_quantum_config = QuantumForgeConfig()
    
    return _global_quantum_config

def initialize_config_system(config_dir: str = "./config") -> QuantumForgeConfig:
    """Initialize QUANTUM-FORGE configuration system."""
    print("  Initializing QUANTUM-FORGE Configuration System...")
    
    # Set config directory
    os.environ['QUANTUM_FORGE_CONFIG_DIR'] = config_dir
    
    # Initialize configuration
    config = get_quantum_config()
    
    # Validate default configuration
    is_valid, errors = config.validate_current_config()
    
    if is_valid:
        print("  Configuration system initialized successfully")
        
        # Show configuration summary
        summary = config.get_config_summary()
        print(f"  Environment: {summary['environment']}")
        print(f"  Current Profile: {summary['current_profile']}")
        print(f"  Available Profiles: {', '.join(summary['available_profiles'])}")
        print(f" ️ Debug Mode: {summary['key_settings']['debug_mode']}")
        print(f"  Log Level: {summary['key_settings']['log_level']}")
        
        return config
    else:
        print("  Configuration validation failed:")
        for error in errors:
            print(f"   {error}")
        return None

# Configuration decorators and utilities
def with_config(profile: str = None):
    """Decorator to inject configuration into function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = get_quantum_config()
            current_profile = profile or config.get_current_profile()
            kwargs['config'] = config.manager.get_config(current_profile)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_config_value(key_path: str, profile: str = None):
    """Decorator to ensure configuration value exists."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = get_quantum_config()
            current_profile = profile or config.get_current_profile()
            value = config.manager.get_config_value(current_profile, key_path)
            
            if value is None:
                raise ValueError(f"Required configuration value missing: {key_path}")
            
            kwargs[key_path.replace('.', '_')] = value
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Initialize configuration system
    config = initialize_config_system("./test_config")
    
    if config:
        print("\n  Testing QUANTUM-FORGE Configuration System...")
        
        # Test profile switching
        print(f"\n  Available profiles: {config.list_profiles()}")
        print(f"Current profile: {config.get_current_profile()}")
        
        # Test configuration access
        trading_config = config.get_trading_config()
        print(f"\n  Trading config sections: {list(trading_config.keys())}")
        
        risk_config = config.get_risk_config()
        print(f" ️ Risk config VaR confidence: {risk_config['var_confidence']}")
        
        # Test convenience methods
        print(f"\n  Debug mode: {config.is_debug_mode()}")
        print(f"  Log level: {config.get_log_level()}")
        print(f"  Max position size: {config.get_max_position_size()}")
        print(f"  Max leverage: {config.get_max_leverage()}")
        print(f" ️ Update frequency: {config.get_update_frequency()}ms")
        
        # Test profile switching
        print(f"\n  Switching to conservative profile...")
        if config.switch_profile('conservative'):
            print(f"New max leverage: {config.get_max_leverage()}")
            print(f"New max position size: {config.get_max_position_size()}")
        
        # Switch back to default
        config.switch_profile('default')
        
        # Test configuration export
        print(f"\n  Exporting configuration...")
        success = config.export_config('default', './test_export.yaml')
        print(f"Export success: {success}")
        
        # Show configuration summary
        print(f"\n  Configuration Summary:")
        summary = config.get_config_summary()
        for key, value in summary.items():
            if key != 'system_info':  # Skip detailed system info
                print(f"   {key}: {value}")
        
        print("\n  Configuration system testing completed!")
    
    else:
        print("  Configuration system initialization failed!")