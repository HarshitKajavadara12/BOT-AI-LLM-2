"""
Toxicity Detection Engine for QUANTUM-FORGE
Implements advanced algorithms to detect toxic flow and informed trading.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Union
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
warnings.filterwarnings('ignore')

class ToxicityLevel(Enum):
    """Toxicity classification levels."""
    BENIGN = 0
    LOW_TOXIC = 1
    MEDIUM_TOXIC = 2
    HIGH_TOXIC = 3
    EXTREMELY_TOXIC = 4

@dataclass
class TradeMetrics:
    """Metrics for individual trades used in toxicity detection."""
    timestamp: float
    price: float
    quantity: int
    side: str  # 'buy' or 'sell'
    trader_id: str
    order_id: str
    market_impact: float
    volume_weighted_price: float
    time_to_execution: float
    order_book_imbalance: float

@dataclass
class ToxicitySignal:
    """Toxicity detection signal."""
    timestamp: float
    trader_id: str
    toxicity_level: ToxicityLevel
    confidence: float
    signal_strength: float
    contributing_factors: Dict[str, float]
    recommended_action: str

class VPINCalculator:
    """Volume-Synchronized Probability of Informed Trading (VPIN) calculator."""
    
    def __init__(self, bucket_size: int = 50000, time_bars: int = 50):
        """
        Initialize VPIN calculator.
        
        Args:
            bucket_size: Volume bucket size for synchronization
            time_bars: Number of time bars for VPIN calculation
        """
        self.bucket_size = bucket_size
        self.time_bars = time_bars
        self.volume_buckets = []
        self.buy_volumes = []
        self.sell_volumes = []
        self.vpin_history = deque(maxlen=1000)
    
    def add_trade(self, volume: int, side: str, price: float):
        """Add trade to VPIN calculation."""
        if not self.volume_buckets or sum(self.volume_buckets) >= self.bucket_size:
            # Start new bucket
            self.volume_buckets = [volume]
            self.buy_volumes.append(volume if side == 'buy' else 0)
            self.sell_volumes.append(volume if side == 'sell' else 0)
        else:
            # Add to current bucket
            self.volume_buckets[-1] += volume
            if side == 'buy':
                self.buy_volumes[-1] += volume
            else:
                self.sell_volumes[-1] += volume
    
    def calculate_vpin(self) -> float:
        """Calculate current VPIN value."""
        if len(self.buy_volumes) < self.time_bars:
            return 0.0
        
        # Get recent buckets
        recent_buy = self.buy_volumes[-self.time_bars:]
        recent_sell = self.sell_volumes[-self.time_bars:]
        
        # Calculate order imbalance for each bucket
        imbalances = []
        for buy_vol, sell_vol in zip(recent_buy, recent_sell):
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                imbalance = abs(buy_vol - sell_vol) / total_vol
                imbalances.append(imbalance)
        
        if not imbalances:
            return 0.0
        
        # VPIN is average order imbalance
        vpin = np.mean(imbalances)
        self.vpin_history.append(vpin)
        
        return vpin
    
    def get_vpin_percentile(self, vpin_value: float) -> float:
        """Get percentile of current VPIN value in historical distribution."""
        if len(self.vpin_history) < 10:
            return 50.0
        
        return stats.percentileofscore(list(self.vpin_history), vpin_value)

class PINModel:
    """Probability of Informed Trading (PIN) model implementation."""
    
    def __init__(self):
        """Initialize PIN model parameters."""
        self.alpha = 0.5    # Probability of information event
        self.delta = 0.5    # Probability of bad news given info event
        self.epsilon_b = 1.0  # Uninformed buy arrival rate
        self.epsilon_s = 1.0  # Uninformed sell arrival rate
        self.mu = 2.0       # Informed trading intensity
        
        self.fitted = False
    
    def fit(self, buy_counts: np.ndarray, sell_counts: np.ndarray):
        """
        Fit PIN model parameters using MLE.
        
        Args:
            buy_counts: Array of buy order counts per time period
            sell_counts: Array of sell order counts per time period
        """
        def negative_log_likelihood(params):
            alpha, delta, eps_b, eps_s, mu = params
            
            # Ensure parameters are positive
            if any(p <= 0 for p in [eps_b, eps_s, mu]) or alpha <= 0 or alpha >= 1 or delta <= 0 or delta >= 1:
                return np.inf
            
            log_likelihood = 0
            
            for b, s in zip(buy_counts, sell_counts):
                # Three possible states: no info, bad news, good news
                
                # No information event
                prob_no_info = (1 - alpha) * stats.poisson.pmf(b, eps_b) * stats.poisson.pmf(s, eps_s)
                
                # Bad news (informed selling)
                prob_bad = alpha * delta * stats.poisson.pmf(b, eps_b) * stats.poisson.pmf(s, eps_s + mu)
                
                # Good news (informed buying)
                prob_good = alpha * (1 - delta) * stats.poisson.pmf(b, eps_b + mu) * stats.poisson.pmf(s, eps_s)
                
                total_prob = prob_no_info + prob_bad + prob_good
                
                if total_prob > 0:
                    log_likelihood += np.log(total_prob)
                else:
                    return np.inf
            
            return -log_likelihood
        
        # Initial parameter guess
        x0 = [0.3, 0.5, np.mean(buy_counts), np.mean(sell_counts), 1.0]
        
        # Bounds for parameters
        bounds = [(0.01, 0.99), (0.01, 0.99), (0.1, None), (0.1, None), (0.1, None)]
        
        # Optimize
        result = minimize(negative_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.alpha, self.delta, self.epsilon_b, self.epsilon_s, self.mu = result.x
            self.fitted = True
        
        return result.success
    
    def calculate_pin(self) -> float:
        """Calculate PIN probability."""
        if not self.fitted:
            return 0.0
        
        numerator = self.alpha * self.mu
        denominator = self.alpha * self.mu + self.epsilon_b + self.epsilon_s
        
        return numerator / denominator

class ToxicityDetector:
    """Advanced toxicity detection system combining multiple algorithms."""
    
    def __init__(self, lookback_window: int = 1000):
        """
        Initialize toxicity detector.
        
        Args:
            lookback_window: Number of trades to consider for analysis
        """
        self.lookback_window = lookback_window
        self.trade_history = deque(maxlen=lookback_window)
        self.trader_profiles = defaultdict(dict)
        
        # Detection models
        self.vpin_calculator = VPINCalculator()
        self.pin_model = PINModel()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Thresholds
        self.vpin_threshold = 0.3
        self.pin_threshold = 0.2
        self.anomaly_threshold = -0.5
        
        self.fitted = False
    
    def add_trade(self, trade_metrics: TradeMetrics):
        """Add trade to toxicity analysis."""
        self.trade_history.append(trade_metrics)
        
        # Update VPIN
        self.vpin_calculator.add_trade(
            trade_metrics.quantity, 
            trade_metrics.side, 
            trade_metrics.price
        )
        
        # Update trader profile
        trader_id = trade_metrics.trader_id
        if trader_id not in self.trader_profiles:
            self.trader_profiles[trader_id] = {
                'trade_count': 0,
                'total_volume': 0,
                'avg_market_impact': 0,
                'trade_frequency': 0,
                'profit_metrics': [],
                'timing_metrics': []
            }
        
        profile = self.trader_profiles[trader_id]
        profile['trade_count'] += 1
        profile['total_volume'] += trade_metrics.quantity
        profile['avg_market_impact'] = (
            (profile['avg_market_impact'] * (profile['trade_count'] - 1) + 
             trade_metrics.market_impact) / profile['trade_count']
        )
        
        # Update timing metrics
        profile['timing_metrics'].append(trade_metrics.time_to_execution)
        if len(profile['timing_metrics']) > 100:
            profile['timing_metrics'].pop(0)
    
    def extract_features(self, trader_id: str, recent_trades: List[TradeMetrics]) -> np.ndarray:
        """Extract features for toxicity detection."""
        if not recent_trades:
            return np.zeros(15)
        
        features = []
        
        # Volume-based features
        volumes = [t.quantity for t in recent_trades]
        features.extend([
            np.mean(volumes),
            np.std(volumes),
            np.max(volumes),
            np.sum(volumes)
        ])
        
        # Market impact features
        impacts = [t.market_impact for t in recent_trades]
        features.extend([
            np.mean(impacts),
            np.std(impacts),
            np.max(impacts)
        ])
        
        # Timing features
        timings = [t.time_to_execution for t in recent_trades]
        features.extend([
            np.mean(timings),
            np.std(timings),
            np.min(timings)
        ])
        
        # Trading pattern features
        buy_trades = sum(1 for t in recent_trades if t.side == 'buy')
        sell_trades = len(recent_trades) - buy_trades
        
        features.extend([
            buy_trades / len(recent_trades) if recent_trades else 0,  # Buy ratio
            len(recent_trades),  # Trade frequency
            np.std([t.price for t in recent_trades]),  # Price volatility
        ])
        
        # Order book features
        imbalances = [t.order_book_imbalance for t in recent_trades]
        features.extend([
            np.mean(imbalances),
            np.std(imbalances)
        ])
        
        return np.array(features)
    
    def fit_models(self):
        """Fit detection models on historical data."""
        if len(self.trade_history) < 100:
            return False
        
        # Prepare data for PIN model
        trades_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'side': t.side,
                'quantity': t.quantity
            }
            for t in self.trade_history
        ])
        
        # Group by time periods (e.g., 5-minute buckets)
        trades_df['time_bucket'] = pd.to_datetime(trades_df['timestamp'], unit='s').dt.floor('5T')
        
        bucket_stats = trades_df.groupby(['time_bucket', 'side'])['quantity'].sum().unstack(fill_value=0)
        
        if 'buy' in bucket_stats.columns and 'sell' in bucket_stats.columns:
            buy_counts = bucket_stats['buy'].values
            sell_counts = bucket_stats['sell'].values
            
            self.pin_model.fit(buy_counts, sell_counts)
        
        # Prepare features for anomaly detection
        all_features = []
        for trader_id in self.trader_profiles.keys():
            trader_trades = [t for t in self.trade_history if t.trader_id == trader_id]
            if len(trader_trades) >= 5:  # Minimum trades for analysis
                features = self.extract_features(trader_id, trader_trades[-50:])  # Recent trades
                all_features.append(features)
        
        if len(all_features) >= 10:
            feature_matrix = np.array(all_features)
            feature_matrix = self.scaler.fit_transform(feature_matrix)
            self.isolation_forest.fit(feature_matrix)
            self.fitted = True
        
        return self.fitted
    
    def detect_toxicity(self, trader_id: str) -> ToxicitySignal:
        """
        Detect toxicity for a specific trader.
        
        Args:
            trader_id: Trader identifier
        
        Returns:
            ToxicitySignal with detection results
        """
        # Get trader's recent trades
        trader_trades = [t for t in self.trade_history if t.trader_id == trader_id]
        
        if len(trader_trades) < 5:
            return ToxicitySignal(
                timestamp=trader_trades[-1].timestamp if trader_trades else 0,
                trader_id=trader_id,
                toxicity_level=ToxicityLevel.BENIGN,
                confidence=0.0,
                signal_strength=0.0,
                contributing_factors={},
                recommended_action="insufficient_data"
            )
        
        contributing_factors = {}
        toxicity_scores = []
        
        # VPIN-based detection
        current_vpin = self.vpin_calculator.calculate_vpin()
        vpin_percentile = self.vpin_calculator.get_vpin_percentile(current_vpin)
        vpin_score = min(1.0, current_vpin / self.vpin_threshold)
        toxicity_scores.append(vpin_score)
        contributing_factors['vpin'] = vpin_score
        
        # PIN-based detection
        pin_value = self.pin_model.calculate_pin()
        pin_score = min(1.0, pin_value / self.pin_threshold)
        toxicity_scores.append(pin_score)
        contributing_factors['pin'] = pin_score
        
        # Anomaly detection
        if self.fitted:
            features = self.extract_features(trader_id, trader_trades[-50:])
            features_scaled = self.scaler.transform([features])
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            
            # Convert to 0-1 scale (more negative = more anomalous)
            normalized_anomaly = max(0, (self.anomaly_threshold - anomaly_score) / abs(self.anomaly_threshold))
            toxicity_scores.append(normalized_anomaly)
            contributing_factors['anomaly'] = normalized_anomaly
        
        # Market impact analysis
        recent_impacts = [t.market_impact for t in trader_trades[-20:]]
        avg_impact = np.mean(recent_impacts) if recent_impacts else 0
        impact_score = min(1.0, avg_impact * 100)  # Scale to 0-1
        toxicity_scores.append(impact_score)
        contributing_factors['market_impact'] = impact_score
        
        # Trading frequency analysis
        if len(trader_trades) >= 2:
            time_diffs = np.diff([t.timestamp for t in trader_trades[-10:]])
            avg_time_between_trades = np.mean(time_diffs) if len(time_diffs) > 0 else 1000
            frequency_score = min(1.0, 60 / max(1, avg_time_between_trades))  # High frequency = higher score
            toxicity_scores.append(frequency_score)
            contributing_factors['frequency'] = frequency_score
        
        # Volume concentration analysis
        volumes = [t.quantity for t in trader_trades[-20:]]
        volume_concentration = np.std(volumes) / max(1, np.mean(volumes)) if volumes else 0
        concentration_score = min(1.0, volume_concentration)
        toxicity_scores.append(concentration_score)
        contributing_factors['volume_concentration'] = concentration_score
        
        # Calculate overall toxicity
        overall_score = np.mean(toxicity_scores) if toxicity_scores else 0
        confidence = min(1.0, len(trader_trades) / 50)  # Confidence based on data availability
        
        # Determine toxicity level
        if overall_score < 0.2:
            toxicity_level = ToxicityLevel.BENIGN
            action = "monitor"
        elif overall_score < 0.4:
            toxicity_level = ToxicityLevel.LOW_TOXIC
            action = "increase_monitoring"
        elif overall_score < 0.6:
            toxicity_level = ToxicityLevel.MEDIUM_TOXIC
            action = "implement_restrictions"
        elif overall_score < 0.8:
            toxicity_level = ToxicityLevel.HIGH_TOXIC
            action = "severe_restrictions"
        else:
            toxicity_level = ToxicityLevel.EXTREMELY_TOXIC
            action = "block_trading"
        
        return ToxicitySignal(
            timestamp=trader_trades[-1].timestamp,
            trader_id=trader_id,
            toxicity_level=toxicity_level,
            confidence=confidence,
            signal_strength=overall_score,
            contributing_factors=contributing_factors,
            recommended_action=action
        )

class NetworkToxicityDetector:
    """Network-based toxicity detection using trader interaction graphs."""
    
    def __init__(self):
        """Initialize network detector."""
        self.trader_network = nx.Graph()
        self.interaction_history = defaultdict(list)
        self.suspicious_clusters = []
    
    def add_interaction(self, trader1: str, trader2: str, interaction_type: str, 
                       timestamp: float, strength: float = 1.0):
        """
        Add trader interaction to network.
        
        Args:
            trader1: First trader
            trader2: Second trader  
            interaction_type: Type of interaction (e.g., 'trade', 'timing', 'pattern')
            timestamp: Interaction timestamp
            strength: Interaction strength
        """
        # Add nodes if not present
        if not self.trader_network.has_node(trader1):
            self.trader_network.add_node(trader1, toxicity_score=0.0)
        if not self.trader_network.has_node(trader2):
            self.trader_network.add_node(trader2, toxicity_score=0.0)
        
        # Add or update edge
        if self.trader_network.has_edge(trader1, trader2):
            # Update existing edge
            current_weight = self.trader_network[trader1][trader2].get('weight', 0)
            self.trader_network[trader1][trader2]['weight'] = current_weight + strength
        else:
            # Add new edge
            self.trader_network.add_edge(trader1, trader2, weight=strength, 
                                       interaction_type=interaction_type)
        
        # Record interaction
        self.interaction_history[(trader1, trader2)].append({
            'timestamp': timestamp,
            'type': interaction_type,
            'strength': strength
        })
    
    def detect_suspicious_clusters(self, min_cluster_size: int = 3, 
                                 weight_threshold: float = 5.0) -> List[List[str]]:
        """
        Detect suspicious trader clusters based on network analysis.
        
        Args:
            min_cluster_size: Minimum size for suspicious cluster
            weight_threshold: Minimum edge weight for inclusion
        
        Returns:
            List of suspicious trader clusters
        """
        # Create subgraph with high-weight edges only
        heavy_edges = [(u, v) for u, v, d in self.trader_network.edges(data=True) 
                      if d['weight'] >= weight_threshold]
        
        subgraph = self.trader_network.edge_subgraph(heavy_edges)
        
        # Find connected components (potential clusters)
        clusters = []
        for component in nx.connected_components(subgraph):
            if len(component) >= min_cluster_size:
                clusters.append(list(component))
        
        # Analyze clusters for suspicious patterns
        suspicious_clusters = []
        for cluster in clusters:
            cluster_subgraph = subgraph.subgraph(cluster)
            
            # Check for high connectivity (potential coordination)
            density = nx.density(cluster_subgraph)
            avg_clustering = nx.average_clustering(cluster_subgraph)
            
            # Suspicious if high density and clustering
            if density > 0.7 and avg_clustering > 0.6:
                suspicious_clusters.append(cluster)
        
        self.suspicious_clusters = suspicious_clusters
        return suspicious_clusters
    
    def calculate_network_toxicity(self, trader_id: str) -> float:
        """
        Calculate network-based toxicity score for trader.
        
        Args:
            trader_id: Trader identifier
        
        Returns:
            Network toxicity score (0-1)
        """
        if not self.trader_network.has_node(trader_id):
            return 0.0
        
        toxicity_components = []
        
        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(self.trader_network)[trader_id]
            closeness = nx.closeness_centrality(self.trader_network)[trader_id]
            degree = nx.degree_centrality(self.trader_network)[trader_id]
            
            # High centrality can indicate suspicious coordination
            centrality_score = (betweenness + closeness + degree) / 3
            toxicity_components.append(centrality_score)
        except:
            toxicity_components.append(0.0)
        
        # Cluster membership
        cluster_score = 0.0
        for cluster in self.suspicious_clusters:
            if trader_id in cluster:
                cluster_score = min(1.0, len(cluster) / 10)  # Larger clusters more suspicious
                break
        toxicity_components.append(cluster_score)
        
        # Edge weight analysis
        edges = self.trader_network.edges(trader_id, data=True)
        edge_weights = [d['weight'] for _, _, d in edges]
        
        if edge_weights:
            max_weight = max(edge_weights)
            avg_weight = np.mean(edge_weights)
            weight_score = min(1.0, (max_weight + avg_weight) / 20)  # Normalize
            toxicity_components.append(weight_score)
        else:
            toxicity_components.append(0.0)
        
        return np.mean(toxicity_components)

