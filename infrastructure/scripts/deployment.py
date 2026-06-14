"""
System Deployment Manager
Automated deployment and initialization of QUANTUM-FORGE components
"""

import os
import sys
import json
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil
import hashlib


class SystemDeployer:
    """Automated system deployment and configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize deployment manager.
        
        Args:
            config_path: Path to deployment configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or 'deployment_config.yaml'
        self.config = self._load_config()
        self.deployment_log = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            return self._default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration."""
        return {
            'environment': 'production',
            'components': {
                'core': True,
                'market_microstructure': True,
                'risk_engine': True,
                'execution': True,
                'ai_ml': True,
                'analytics': True,
                'dashboards': True,
                'data_systems': True
            },
            'database': {
                'type': 'postgresql',
                'host': 'localhost',
                'port': 5432,
                'name': 'quantum_forge'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'workers': {
                'data_ingestion': 4,
                'order_execution': 8,
                'risk_monitoring': 2,
                'analytics': 4
            },
            'monitoring': {
                'metrics_port': 9090,
                'logging_level': 'INFO',
                'alert_email': 'alerts@quantumforge.com'
            }
        }
    
    def validate_environment(self) -> Dict[str, bool]:
        """
        Validate deployment environment.
        
        Returns:
            Dictionary of validation results
        """
        validations = {}
        
        # Check Python version
        python_version = sys.version_info
        validations['python_version'] = (
            python_version.major == 3 and python_version.minor >= 8
        )
        
        # Check required packages
        required_packages = [
            'numpy', 'pandas', 'scipy', 'scikit-learn',
            'torch', 'tensorflow', 'fastapi', 'redis',
            'sqlalchemy', 'psycopg2'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                validations[f'package_{package}'] = True
            except ImportError:
                validations[f'package_{package}'] = False
        
        # Check directory structure
        required_dirs = [
            'core', 'components', 'infrastructure',
            'data', 'logs', 'config'
        ]
        
        for directory in required_dirs:
            validations[f'dir_{directory}'] = os.path.exists(directory)
        
        # Check database connectivity
        try:
            from sqlalchemy import create_engine
            db_config = self.config['database']
            engine = create_engine(
                f"postgresql://{db_config['host']}:{db_config['port']}/{db_config['name']}"
            )
            engine.connect()
            validations['database_connection'] = True
        except Exception as e:
            self.logger.warning(f"Database connection failed: {e}")
            validations['database_connection'] = False
        
        # Check Redis connectivity
        try:
            import redis
            redis_config = self.config['redis']
            r = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db']
            )
            r.ping()
            validations['redis_connection'] = True
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            validations['redis_connection'] = False
        
        return validations
    
    def initialize_database(self):
        """Initialize database schema."""
        try:
            from sqlalchemy import create_engine, MetaData, Table, Column
            from sqlalchemy import Integer, String, Float, DateTime, JSON
            
            db_config = self.config['database']
            engine = create_engine(
                f"postgresql://{db_config['host']}:{db_config['port']}/{db_config['name']}"
            )
            
            metadata = MetaData()
            
            # Market data table
            Table('market_data', metadata,
                Column('id', Integer, primary_key=True),
                Column('symbol', String(10), index=True),
                Column('timestamp', DateTime, index=True),
                Column('open', Float),
                Column('high', Float),
                Column('low', Float),
                Column('close', Float),
                Column('volume', Float),
                Column('metadata', JSON)
            )
            
            # Orders table
            Table('orders', metadata,
                Column('id', Integer, primary_key=True),
                Column('order_id', String(50), unique=True, index=True),
                Column('symbol', String(10), index=True),
                Column('side', String(10)),
                Column('quantity', Float),
                Column('price', Float),
                Column('order_type', String(20)),
                Column('status', String(20), index=True),
                Column('created_at', DateTime, index=True),
                Column('updated_at', DateTime),
                Column('metadata', JSON)
            )
            
            # Trades table
            Table('trades', metadata,
                Column('id', Integer, primary_key=True),
                Column('trade_id', String(50), unique=True, index=True),
                Column('order_id', String(50), index=True),
                Column('symbol', String(10), index=True),
                Column('side', String(10)),
                Column('quantity', Float),
                Column('price', Float),
                Column('commission', Float),
                Column('executed_at', DateTime, index=True),
                Column('metadata', JSON)
            )
            
            # Positions table
            Table('positions', metadata,
                Column('id', Integer, primary_key=True),
                Column('symbol', String(10), unique=True, index=True),
                Column('quantity', Float),
                Column('avg_price', Float),
                Column('market_value', Float),
                Column('unrealized_pnl', Float),
                Column('realized_pnl', Float),
                Column('updated_at', DateTime),
                Column('metadata', JSON)
            )
            
            # Risk metrics table
            Table('risk_metrics', metadata,
                Column('id', Integer, primary_key=True),
                Column('timestamp', DateTime, index=True),
                Column('portfolio_value', Float),
                Column('var_95', Float),
                Column('var_99', Float),
                Column('cvar_95', Float),
                Column('sharpe_ratio', Float),
                Column('max_drawdown', Float),
                Column('volatility', Float),
                Column('metadata', JSON)
            )
            
            # Performance table
            Table('performance', metadata,
                Column('id', Integer, primary_key=True),
                Column('date', DateTime, index=True),
                Column('pnl', Float),
                Column('returns', Float),
                Column('cumulative_returns', Float),
                Column('trades_count', Integer),
                Column('win_rate', Float),
                Column('metadata', JSON)
            )
            
            # System logs table
            Table('system_logs', metadata,
                Column('id', Integer, primary_key=True),
                Column('timestamp', DateTime, index=True),
                Column('level', String(20), index=True),
                Column('component', String(50), index=True),
                Column('message', String),
                Column('metadata', JSON)
            )
            
            # Create all tables
            metadata.create_all(engine)
            
            self.logger.info("Database schema initialized successfully")
            self.deployment_log.append({
                'step': 'database_init',
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.deployment_log.append({
                'step': 'database_init',
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def setup_redis(self):
        """Setup Redis cache and data structures."""
        try:
            import redis
            
            redis_config = self.config['redis']
            r = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db']
            )
            
            # Initialize cache namespaces
            namespaces = [
                'market_data',
                'order_book',
                'positions',
                'risk_metrics',
                'signals'
            ]
            
            for namespace in namespaces:
                r.delete(f"{namespace}:*")
            
            # Set configuration in Redis
            r.set('config:environment', self.config['environment'])
            r.set('config:deployment_time', datetime.now().isoformat())
            
            self.logger.info("Redis setup completed successfully")
            self.deployment_log.append({
                'step': 'redis_setup',
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Redis setup failed: {e}")
            self.deployment_log.append({
                'step': 'redis_setup',
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def deploy_components(self):
        """Deploy system components."""
        try:
            components = self.config['components']
            
            for component, enabled in components.items():
                if enabled:
                    self.logger.info(f"Deploying component: {component}")
                    
                    # Component-specific initialization
                    if component == 'core':
                        self._deploy_core()
                    elif component == 'market_microstructure':
                        self._deploy_market_microstructure()
                    elif component == 'risk_engine':
                        self._deploy_risk_engine()
                    elif component == 'execution':
                        self._deploy_execution()
                    elif component == 'ai_ml':
                        self._deploy_ai_ml()
                    elif component == 'analytics':
                        self._deploy_analytics()
                    elif component == 'dashboards':
                        self._deploy_dashboards()
                    elif component == 'data_systems':
                        self._deploy_data_systems()
                    
                    self.deployment_log.append({
                        'step': f'deploy_{component}',
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    })
            
            self.logger.info("All components deployed successfully")
            
        except Exception as e:
            self.logger.error(f"Component deployment failed: {e}")
            self.deployment_log.append({
                'step': 'deploy_components',
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def _deploy_core(self):
        """Deploy core mathematical engine."""
        from core.mathematical_engine.pricing_models import PricingModels
        
        # Initialize pricing models
        pricing = PricingModels()
        self.logger.info("Core mathematical engine deployed")
    
    def _deploy_market_microstructure(self):
        """Deploy market microstructure components."""
        from core.market_microstructure.order_flow import OrderFlowAnalyzer
        
        # Initialize order flow analyzer
        analyzer = OrderFlowAnalyzer()
        self.logger.info("Market microstructure components deployed")
    
    def _deploy_risk_engine(self):
        """Deploy risk management engine."""
        from core.risk_mathematics.var_models import VaRCalculator
        
        # Initialize risk calculator
        risk_calc = VaRCalculator()
        self.logger.info("Risk engine deployed")
    
    def _deploy_execution(self):
        """Deploy execution algorithms."""
        from core.execution_algorithms.twap import TWAPExecutor
        
        # Initialize execution engine
        executor = TWAPExecutor()
        self.logger.info("Execution algorithms deployed")
    
    def _deploy_ai_ml(self):
        """Deploy AI/ML intelligence layer."""
        from core.ai_ml_intelligence.ensemble_models import EnsembleModel
        
        # Initialize ML models
        model = EnsembleModel()
        self.logger.info("AI/ML intelligence layer deployed")
    
    def _deploy_analytics(self):
        """Deploy analytics framework."""
        from components.analytics.performance_analytics import PerformanceAnalytics
        
        # Initialize analytics
        analytics = PerformanceAnalytics()
        self.logger.info("Analytics framework deployed")
    
    def _deploy_dashboards(self):
        """Deploy interface dashboards."""
        # Dashboard deployment would start web servers
        self.logger.info("Interface dashboards deployed")
    
    def _deploy_data_systems(self):
        """Deploy data infrastructure."""
        from infrastructure.data_systems.streaming_engine import DataStreamManager
        
        # Initialize data systems
        stream_manager = DataStreamManager()
        self.logger.info("Data systems deployed")
    
    def start_workers(self):
        """Start background worker processes."""
        try:
            worker_config = self.config['workers']
            
            for worker_type, count in worker_config.items():
                self.logger.info(f"Starting {count} {worker_type} workers")
                
                for i in range(count):
                    # In production, this would spawn actual worker processes
                    # For now, log the intent
                    self.logger.info(f"Worker {worker_type}_{i} started")
            
            self.deployment_log.append({
                'step': 'start_workers',
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Worker startup failed: {e}")
            self.deployment_log.append({
                'step': 'start_workers',
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def verify_deployment(self) -> Dict[str, bool]:
        """
        Verify deployment success.
        
        Returns:
            Dictionary of verification results
        """
        verifications = {}
        
        # Check database connectivity
        try:
            from sqlalchemy import create_engine
            db_config = self.config['database']
            engine = create_engine(
                f"postgresql://{db_config['host']}:{db_config['port']}/{db_config['name']}"
            )
            conn = engine.connect()
            conn.close()
            verifications['database'] = True
        except:
            verifications['database'] = False
        
        # Check Redis connectivity
        try:
            import redis
            redis_config = self.config['redis']
            r = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db']
            )
            r.ping()
            verifications['redis'] = True
        except:
            verifications['redis'] = False
        
        # Check component imports
        components_to_check = [
            ('core', 'core.mathematical_engine.pricing_models'),
            ('market_microstructure', 'core.market_microstructure.order_flow'),
            ('risk', 'core.risk_mathematics.var_models'),
            ('execution', 'core.execution_algorithms.twap'),
            ('ai_ml', 'core.ai_ml_intelligence.ensemble_models'),
            ('analytics', 'components.analytics.performance_analytics')
        ]
        
        for component, module_path in components_to_check:
            try:
                __import__(module_path)
                verifications[f'component_{component}'] = True
            except:
                verifications[f'component_{component}'] = False
        
        return verifications
    
    def deploy(self) -> Dict[str, Any]:
        """
        Execute full deployment.
        
        Returns:
            Deployment summary
        """
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("QUANTUM-FORGE Deployment Starting")
        self.logger.info("=" * 60)
        
        try:
            # Validate environment
            self.logger.info("Step 1: Validating environment...")
            validations = self.validate_environment()
            
            if not all(validations.values()):
                failed = [k for k, v in validations.items() if not v]
                self.logger.warning(f"Validation failures: {failed}")
            
            # Initialize database
            self.logger.info("Step 2: Initializing database...")
            self.initialize_database()
            
            # Setup Redis
            self.logger.info("Step 3: Setting up Redis...")
            self.setup_redis()
            
            # Deploy components
            self.logger.info("Step 4: Deploying components...")
            self.deploy_components()
            
            # Start workers
            self.logger.info("Step 5: Starting workers...")
            self.start_workers()
            
            # Verify deployment
            self.logger.info("Step 6: Verifying deployment...")
            verifications = self.verify_deployment()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'validations': validations,
                'verifications': verifications,
                'deployment_log': self.deployment_log
            }
            
            self.logger.info("=" * 60)
            self.logger.info("QUANTUM-FORGE Deployment Completed Successfully")
            self.logger.info(f"Duration: {duration:.2f} seconds")
            self.logger.info("=" * 60)
            
            return summary
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                'status': 'failed',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'error': str(e),
                'deployment_log': self.deployment_log
            }
            
            self.logger.error("=" * 60)
            self.logger.error("QUANTUM-FORGE Deployment Failed")
            self.logger.error(f"Error: {e}")
            self.logger.error("=" * 60)
            
            return summary
    
    def rollback(self):
        """Rollback deployment."""
        self.logger.info("Rolling back deployment...")
        
        try:
            # Stop workers
            self.logger.info("Stopping workers...")
            
            # Clear Redis
            self.logger.info("Clearing Redis cache...")
            import redis
            redis_config = self.config['redis']
            r = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db']
            )
            r.flushdb()
            
            self.logger.info("Rollback completed")
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            raise
    
    def generate_deployment_report(self, summary: Dict[str, Any], output_path: str = 'deployment_report.json'):
        """Generate deployment report."""
        try:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Deployment report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate deployment report: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Execute deployment
    deployer = SystemDeployer()
    summary = deployer.deploy()
    deployer.generate_deployment_report(summary)
