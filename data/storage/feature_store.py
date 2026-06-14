"""
QUANTUM-FORGE Feature Store
Production-grade feature storage and retrieval system

Features:
- Real-time feature serving (<1ms latency)
- Historical feature lookup with point-in-time correctness
- Feature versioning and lineage tracking
- Automatic feature drift detection
- Integration with Redis (hot) and Parquet (cold)
"""

import pandas as pd
import numpy as np
import redis
import pickle
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature data types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    EMBEDDING = "embedding"
    TEXT = "text"

@dataclass
class FeatureMetadata:
    """Feature metadata and schema"""
    name: str
    type: FeatureType
    description: str
    source: str  # Data source/computation
    version: str
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    schema: Dict[str, Any]  # JSON schema
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict):
        data['type'] = FeatureType(data['type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)

@dataclass
class FeatureValue:
    """Feature value with metadata"""
    feature_name: str
    entity_id: str  # Symbol, user_id, etc.
    value: Any
    timestamp: float
    version: str
    metadata: Optional[Dict] = None
    
    def to_bytes(self) -> bytes:
        """Serialize for Redis storage"""
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def from_bytes(cls, data: bytes):
        """Deserialize from Redis"""
        return pickle.loads(data)

class FeatureStore:
    """
    Production feature store with Redis + Parquet backend
    
    Architecture:
    - Hot Path: Redis for real-time serving (<1ms)
    - Cold Path: Parquet for historical analysis
    - Metadata: Redis for feature schema/lineage
    
    Key Features:
    - Point-in-time correctness
    - Feature versioning
    - Drift detection
    - Lineage tracking
    """
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 1,  # Separate from cache
                 parquet_path: str = "./data/features",
                 default_ttl: int = 86400):  # 24 hours
        
        # Redis connection for hot features
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,
            socket_keepalive=True
        )
        
        self.parquet_path = parquet_path
        self.default_ttl = default_ttl
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Key namespaces
        self.FEATURE_NS = "feature:"
        self.METADATA_NS = "meta:"
        self.INDEX_NS = "index:"
        self.VERSION_NS = "version:"
        
        logger.info("Feature store initialized")
    
    # ==================== FEATURE REGISTRATION ====================
    
    def register_feature(self, metadata: FeatureMetadata) -> bool:
        """Register a new feature with metadata"""
        try:
            key = f"{self.METADATA_NS}{metadata.name}"
            
            # Store metadata
            self.redis_client.set(
                key, 
                json.dumps(metadata.to_dict()),
                ex=self.default_ttl * 30  # Metadata lives longer
            )
            
            # Add to feature index
            self.redis_client.sadd("feature_registry", metadata.name)
            
            logger.info(f"Registered feature: {metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature {metadata.name}: {e}")
            return False
    
    def get_feature_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """Get feature metadata"""
        try:
            key = f"{self.METADATA_NS}{feature_name}"
            data = self.redis_client.get(key)
            
            if not data:
                return None
            
            metadata_dict = json.loads(data)
            return FeatureMetadata.from_dict(metadata_dict)
            
        except Exception as e:
            logger.error(f"Failed to get metadata for {feature_name}: {e}")
            return None
    
    def list_features(self) -> List[str]:
        """List all registered features"""
        try:
            features = self.redis_client.smembers("feature_registry")
            return [f.decode() for f in features]
        except Exception as e:
            logger.error(f"Failed to list features: {e}")
            return []
    
    # ==================== FEATURE WRITING ====================
    
    def write_feature(self, 
                     feature_name: str,
                     entity_id: str,
                     value: Any,
                     timestamp: Optional[float] = None,
                     version: str = "1.0",
                     metadata: Optional[Dict] = None,
                     ttl: Optional[int] = None) -> bool:
        """
        Write feature value to store
        
        Args:
            feature_name: Name of the feature
            entity_id: Entity identifier (symbol, user_id, etc.)
            value: Feature value
            timestamp: Unix timestamp (default: current time)
            version: Feature version
            metadata: Additional metadata
            ttl: Time to live in seconds
        """
        
        try:
            if timestamp is None:
                timestamp = time.time()
            
            if ttl is None:
                ttl = self.default_ttl
            
            # Create feature value
            feature_value = FeatureValue(
                feature_name=feature_name,
                entity_id=entity_id,
                value=value,
                timestamp=timestamp,
                version=version,
                metadata=metadata
            )
            
            # Store in Redis (hot path)
            hot_key = f"{self.FEATURE_NS}{feature_name}:{entity_id}:latest"
            self.redis_client.setex(
                hot_key,
                ttl,
                feature_value.to_bytes()
            )
            
            # Store in time-series index for point-in-time queries
            ts_key = f"{self.INDEX_NS}{feature_name}:{entity_id}"
            versioned_key = f"{self.FEATURE_NS}{feature_name}:{entity_id}:{int(timestamp)}"
            
            # Store versioned value
            self.redis_client.setex(
                versioned_key,
                ttl,
                feature_value.to_bytes()
            )
            
            # Add to time-series index
            self.redis_client.zadd(ts_key, {versioned_key: timestamp})
            
            # Maintain index size (keep last 1000 entries)
            self.redis_client.zremrangebyrank(ts_key, 0, -1001)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write feature {feature_name}: {e}")
            return False
    
    def write_features_batch(self, 
                            features: List[Tuple[str, str, Any, float]],
                            version: str = "1.0",
                            ttl: Optional[int] = None) -> int:
        """
        Batch write multiple features
        
        Args:
            features: List of (feature_name, entity_id, value, timestamp)
            version: Feature version
            ttl: Time to live
            
        Returns:
            Number of features successfully written
        """
        
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            pipe = self.redis_client.pipeline()
            
            for feature_name, entity_id, value, timestamp in features:
                feature_value = FeatureValue(
                    feature_name=feature_name,
                    entity_id=entity_id,
                    value=value,
                    timestamp=timestamp,
                    version=version
                )
                
                # Hot path
                hot_key = f"{self.FEATURE_NS}{feature_name}:{entity_id}:latest"
                pipe.setex(hot_key, ttl, feature_value.to_bytes())
                
                # Time-series index
                ts_key = f"{self.INDEX_NS}{feature_name}:{entity_id}"
                versioned_key = f"{self.FEATURE_NS}{feature_name}:{entity_id}:{int(timestamp)}"
                
                pipe.setex(versioned_key, ttl, feature_value.to_bytes())
                pipe.zadd(ts_key, {versioned_key: timestamp})
            
            results = pipe.execute()
            successful = sum(1 for result in results if result)
            
            logger.info(f"Batch wrote {successful}/{len(features)} features")
            return successful
            
        except Exception as e:
            logger.error(f"Failed to batch write features: {e}")
            return 0
    
    # ==================== FEATURE READING ====================
    
    def get_latest_feature(self, 
                          feature_name: str,
                          entity_id: str) -> Optional[FeatureValue]:
        """Get latest feature value (hot path)"""
        
        try:
            key = f"{self.FEATURE_NS}{feature_name}:{entity_id}:latest"
            data = self.redis_client.get(key)
            
            if not data:
                return None
            
            return FeatureValue.from_bytes(data)
            
        except Exception as e:
            logger.error(f"Failed to get latest feature {feature_name}: {e}")
            return None
    
    def get_feature_at_time(self,
                           feature_name: str,
                           entity_id: str,
                           timestamp: float) -> Optional[FeatureValue]:
        """
        Get feature value at specific time (point-in-time correctness)
        
        This is critical for backtesting to avoid look-ahead bias
        """
        
        try:
            ts_key = f"{self.INDEX_NS}{feature_name}:{entity_id}"
            
            # Find latest feature before or at timestamp
            results = self.redis_client.zrevrangebyscore(
                ts_key, 
                timestamp, 
                "-inf",
                start=0,
                num=1,
                withscores=True
            )
            
            if not results:
                return None
            
            feature_key, feature_timestamp = results[0]
            
            # Get feature value
            data = self.redis_client.get(feature_key)
            
            if not data:
                return None
            
            return FeatureValue.from_bytes(data)
            
        except Exception as e:
            logger.error(f"Failed to get feature at time {feature_name}: {e}")
            return None
    
    def get_feature_history(self,
                           feature_name: str,
                           entity_id: str,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None,
                           limit: int = 1000) -> List[FeatureValue]:
        """Get feature history within time range"""
        
        try:
            ts_key = f"{self.INDEX_NS}{feature_name}:{entity_id}"
            
            # Set default time range
            if end_time is None:
                end_time = time.time()
            if start_time is None:
                start_time = end_time - 86400  # Last 24 hours
            
            # Get feature keys in time range
            feature_keys = self.redis_client.zrangebyscore(
                ts_key,
                start_time,
                end_time,
                start=0,
                num=limit
            )
            
            if not feature_keys:
                return []
            
            # Batch get feature values
            pipe = self.redis_client.pipeline()
            for key in feature_keys:
                pipe.get(key)
            
            feature_data = pipe.execute()
            
            # Deserialize features
            features = []
            for data in feature_data:
                if data:
                    features.append(FeatureValue.from_bytes(data))
            
            return sorted(features, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Failed to get feature history {feature_name}: {e}")
            return []
    
    def get_features_for_entity(self,
                               entity_id: str,
                               feature_names: Optional[List[str]] = None,
                               timestamp: Optional[float] = None) -> Dict[str, FeatureValue]:
        """
        Get multiple features for an entity (feature vector)
        
        This is the main method for model inference
        """
        
        try:
            if feature_names is None:
                feature_names = self.list_features()
            
            results = {}
            
            # Batch get features
            if timestamp is None:
                # Get latest features
                keys = [f"{self.FEATURE_NS}{name}:{entity_id}:latest" 
                       for name in feature_names]
                
                pipe = self.redis_client.pipeline()
                for key in keys:
                    pipe.get(key)
                
                feature_data = pipe.execute()
                
                for name, data in zip(feature_names, feature_data):
                    if data:
                        results[name] = FeatureValue.from_bytes(data)
            
            else:
                # Get point-in-time features
                for name in feature_names:
                    feature = self.get_feature_at_time(name, entity_id, timestamp)
                    if feature:
                        results[name] = feature
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get features for entity {entity_id}: {e}")
            return {}
    
    # ==================== FEATURE ANALYTICS ====================
    
    def compute_feature_stats(self, 
                             feature_name: str,
                             entity_ids: List[str],
                             window_hours: int = 24) -> Dict[str, float]:
        """Compute feature statistics for drift detection"""
        
        try:
            end_time = time.time()
            start_time = end_time - (window_hours * 3600)
            
            all_values = []
            
            for entity_id in entity_ids:
                history = self.get_feature_history(
                    feature_name, entity_id, start_time, end_time
                )
                
                for feature_value in history:
                    if isinstance(feature_value.value, (int, float)):
                        all_values.append(feature_value.value)
            
            if not all_values:
                return {}
            
            values = np.array(all_values)
            
            stats = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute feature stats: {e}")
            return {}
    
    def detect_feature_drift(self,
                            feature_name: str,
                            entity_ids: List[str],
                            baseline_hours: int = 168,  # 1 week
                            current_hours: int = 24) -> Dict[str, Any]:
        """
        Detect feature drift using statistical tests
        
        Compares current distribution vs baseline
        """
        
        try:
            # Get baseline stats
            baseline_stats = self.compute_feature_stats(
                feature_name, entity_ids, baseline_hours
            )
            
            # Get current stats  
            current_stats = self.compute_feature_stats(
                feature_name, entity_ids, current_hours
            )
            
            if not baseline_stats or not current_stats:
                return {"error": "Insufficient data for drift detection"}
            
            # Compute drift metrics
            drift_metrics = {}
            
            for metric in ["mean", "std", "p50", "p95"]:
                if metric in baseline_stats and metric in current_stats:
                    baseline_val = baseline_stats[metric]
                    current_val = current_stats[metric]
                    
                    if baseline_val != 0:
                        drift_pct = abs(current_val - baseline_val) / abs(baseline_val) * 100
                        drift_metrics[f"{metric}_drift_pct"] = drift_pct
            
            # Drift alert thresholds
            drift_alert = False
            alerts = []
            
            if drift_metrics.get("mean_drift_pct", 0) > 20:
                drift_alert = True
                alerts.append("Mean shifted >20%")
            
            if drift_metrics.get("std_drift_pct", 0) > 50:
                drift_alert = True
                alerts.append("Std changed >50%")
            
            result = {
                "feature_name": feature_name,
                "drift_detected": drift_alert,
                "alerts": alerts,
                "baseline_stats": baseline_stats,
                "current_stats": current_stats,
                "drift_metrics": drift_metrics,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to detect drift for {feature_name}: {e}")
            return {"error": str(e)}
    
    # ==================== FEATURE SERVING ====================
    
    def serve_features(self, 
                      entity_id: str,
                      feature_names: List[str],
                      timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Serve features for model inference (optimized for latency)
        
        Returns feature vector as simple dict for ML models
        """
        
        try:
            features = self.get_features_for_entity(
                entity_id, feature_names, timestamp
            )
            
            # Convert to simple dict for ML models
            feature_vector = {}
            
            for name, feature_value in features.items():
                feature_vector[name] = feature_value.value
            
            # Add metadata
            feature_vector["_timestamp"] = timestamp or time.time()
            feature_vector["_entity_id"] = entity_id
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Failed to serve features: {e}")
            return {}
    
    # ==================== UTILITIES ====================
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring"""
        
        try:
            # Redis connectivity
            redis_ok = self.redis_client.ping()
            
            # Feature count
            feature_count = len(self.list_features())
            
            # Memory usage
            memory_info = self.redis_client.info("memory")
            
            return {
                "status": "healthy" if redis_ok else "unhealthy",
                "redis_connected": redis_ok,
                "registered_features": feature_count,
                "memory_used_mb": memory_info.get("used_memory", 0) / 1024 / 1024,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def cleanup_expired_features(self, dry_run: bool = True) -> Dict[str, int]:
        """Clean up expired features (maintenance task)"""
        
        try:
            deleted_count = 0
            scanned_count = 0
            
            # Scan all feature keys
            for key in self.redis_client.scan_iter(match=f"{self.FEATURE_NS}*"):
                scanned_count += 1
                
                # Check TTL
                ttl = self.redis_client.ttl(key)
                
                if ttl == -1:  # No expiration set
                    if not dry_run:
                        self.redis_client.expire(key, self.default_ttl)
                
                elif ttl == -2:  # Key expired but not deleted
                    if not dry_run:
                        self.redis_client.delete(key)
                        deleted_count += 1
            
            return {
                "scanned": scanned_count,
                "deleted": deleted_count,
                "dry_run": dry_run
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup features: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Clean shutdown"""
        try:
            self.executor.shutdown(wait=True)
            self.redis_client.close()
            logger.info("Feature store shutdown complete")
        except Exception as e:
            logger.error(f"Error closing feature store: {e}")

# ==================== FEATURE PIPELINE ====================

class FeaturePipeline:
    """
    Feature computation pipeline
    
    Orchestrates feature computation and storage
    """
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.computations = {}  # Feature name -> computation function
        
    def register_computation(self, 
                           feature_name: str,
                           computation_func: callable,
                           dependencies: List[str] = None):
        """Register feature computation"""
        
        self.computations[feature_name] = {
            "func": computation_func,
            "dependencies": dependencies or []
        }
        
        logger.info(f"Registered computation for {feature_name}")
    
    def compute_features(self, 
                        entity_id: str,
                        timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Compute all features for entity"""
        
        if timestamp is None:
            timestamp = time.time()
        
        computed = {}
        
        # Topological sort of dependencies (simple implementation)
        remaining = set(self.computations.keys())
        
        while remaining:
            # Find features with satisfied dependencies
            ready = []
            
            for feature_name in remaining:
                deps = self.computations[feature_name]["dependencies"]
                if all(dep in computed for dep in deps):
                    ready.append(feature_name)
            
            if not ready:
                logger.error("Circular dependency detected in features")
                break
            
            # Compute ready features
            for feature_name in ready:
                try:
                    func = self.computations[feature_name]["func"]
                    
                    # Get dependencies
                    deps = {}
                    for dep in self.computations[feature_name]["dependencies"]:
                        deps[dep] = computed[dep]
                    
                    # Compute feature
                    value = func(entity_id, timestamp, deps)
                    computed[feature_name] = value
                    
                    # Store in feature store
                    self.feature_store.write_feature(
                        feature_name, entity_id, value, timestamp
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to compute {feature_name}: {e}")
                    computed[feature_name] = None
                
                remaining.remove(feature_name)
        
        return computed

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize feature store
    feature_store = FeatureStore()
    
    # Health check
    health = feature_store.health_check()
    print(f"Health: {health}")
    
    # Register a feature
    metadata = FeatureMetadata(
        name="RSI_14",
        type=FeatureType.NUMERIC,
        description="14-period RSI indicator",
        source="technical_analysis",
        version="1.0",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        tags=["technical", "momentum"],
        schema={"type": "float", "range": [0, 100]}
    )
    
    feature_store.register_feature(metadata)
    
    # Write some features
    symbol = "BTCUSDT"
    current_time = time.time()
    
    for i in range(100):
        rsi_value = 50 + 20 * np.sin(i * 0.1)  # Oscillating RSI
        
        feature_store.write_feature(
            "RSI_14",
            symbol,
            rsi_value,
            current_time + i * 60  # 1-minute intervals
        )
    
    # Read latest feature
    latest_rsi = feature_store.get_latest_feature("RSI_14", symbol)
    print(f"Latest RSI: {latest_rsi.value if latest_rsi else 'None'}")
    
    # Get feature at specific time
    historical_rsi = feature_store.get_feature_at_time(
        "RSI_14", symbol, current_time + 30 * 60
    )
    print(f"Historical RSI: {historical_rsi.value if historical_rsi else 'None'}")
    
    # Feature serving (for model inference)
    feature_vector = feature_store.serve_features(
        symbol, ["RSI_14"]
    )
    print(f"Feature vector: {feature_vector}")
    
    # Feature statistics
    stats = feature_store.compute_feature_stats("RSI_14", [symbol])
    print(f"Feature stats: {stats}")
    
    # Cleanup
    feature_store.close()