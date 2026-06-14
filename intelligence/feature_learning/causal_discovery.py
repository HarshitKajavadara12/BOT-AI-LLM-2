"""
Causal Discovery for Financial Markets
Methods for discovering causal relationships in financial time series and market data
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union
from abc import ABC, abstractmethod
import networkx as nx
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
import warnings
from itertools import combinations, permutations


class CausalDiscoveryMethod(ABC):
    """Abstract base class for causal discovery methods"""
    
    @abstractmethod
    def discover_graph(self, X: np.ndarray, variable_names: Optional[List[str]] = None) -> nx.DiGraph:
        """
        Discover causal graph from data
        
        Args:
            X: Data matrix [n_samples, n_variables]
            variable_names: Names of variables
        
        Returns:
            Directed graph representing causal relationships
        """
        pass
    
    @abstractmethod
    def estimate_causal_strength(self, X: np.ndarray, cause: int, effect: int) -> float:
        """
        Estimate strength of causal relationship
        
        Args:
            X: Data matrix
            cause: Index of cause variable
            effect: Index of effect variable
        
        Returns:
            Causal strength estimate
        """
        pass


class PCAlgorithm(CausalDiscoveryMethod):
    """
    PC Algorithm for causal discovery
    Based on conditional independence testing
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        max_conditioning_set_size: int = 3,
        independence_test: str = 'correlation'
    ):
        self.alpha = alpha
        self.max_conditioning_set_size = max_conditioning_set_size
        self.independence_test = independence_test
    
    def _test_independence(
        self,
        X: np.ndarray,
        i: int,
        j: int,
        conditioning_set: List[int]
    ) -> bool:
        """Test conditional independence between variables i and j given conditioning set"""
        
        if len(conditioning_set) == 0:
            # Marginal independence
            if self.independence_test == 'correlation':
                corr, p_value = stats.pearsonr(X[:, i], X[:, j])
                return p_value > self.alpha
            elif self.independence_test == 'mutual_info':
                from sklearn.feature_selection import mutual_info_regression
                mi = mutual_info_regression(X[:, i].reshape(-1, 1), X[:, j])
                # Approximate p-value using chi-square distribution
                test_stat = 2 * len(X) * mi[0]
                p_value = 1 - stats.chi2.cdf(test_stat, df=1)
                return p_value > self.alpha
        else:
            # Conditional independence using partial correlation
            data_subset = X[:, [i, j] + conditioning_set]
            
            try:
                # Compute partial correlation
                partial_corr = self._partial_correlation(data_subset, 0, 1, list(range(2, data_subset.shape[1])))
                
                # Test significance
                n = len(X)
                k = len(conditioning_set)
                df = n - k - 2
                
                if df <= 0:
                    return True  # Cannot test, assume independent
                
                t_stat = partial_corr * np.sqrt(df / (1 - partial_corr**2 + 1e-10))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                return p_value > self.alpha
                
            except np.linalg.LinAlgError:
                return True  # Assume independent if computation fails
    
    def _partial_correlation(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        conditioning_vars: List[int]
    ) -> float:
        """Compute partial correlation between variables i and j given conditioning variables"""
        
        if len(conditioning_vars) == 0:
            return np.corrcoef(data[:, i], data[:, j])[0, 1]
        
        # Regress out conditioning variables
        X_cond = data[:, conditioning_vars]
        
        # Residuals after regressing on conditioning variables
        reg_i = LinearRegression().fit(X_cond, data[:, i])
        res_i = data[:, i] - reg_i.predict(X_cond)
        
        reg_j = LinearRegression().fit(X_cond, data[:, j])
        res_j = data[:, j] - reg_j.predict(X_cond)
        
        # Correlation of residuals
        return np.corrcoef(res_i, res_j)[0, 1]
    
    def discover_graph(self, X: np.ndarray, variable_names: Optional[List[str]] = None) -> nx.DiGraph:
        """Discover causal graph using PC algorithm"""
        
        n_vars = X.shape[1]
        
        if variable_names is None:
            variable_names = [f"X{i}" for i in range(n_vars)]
        
        # Start with complete undirected graph
        graph = nx.Graph()
        graph.add_nodes_from(variable_names)
        
        # Add all possible edges
        for i, j in combinations(range(n_vars), 2):
            graph.add_edge(variable_names[i], variable_names[j])
        
        # PC algorithm: iteratively remove edges based on conditional independence
        for conditioning_size in range(self.max_conditioning_set_size + 1):
            edges_to_remove = []
            
            for i, j in combinations(range(n_vars), 2):
                var_i, var_j = variable_names[i], variable_names[j]
                
                if not graph.has_edge(var_i, var_j):
                    continue
                
                # Find potential conditioning sets
                neighbors_i = [variable_names[k] for k in range(n_vars) 
                              if k != i and k != j and graph.has_edge(var_i, variable_names[k])]
                neighbors_j = [variable_names[k] for k in range(n_vars) 
                              if k != i and k != j and graph.has_edge(var_j, variable_names[k])]
                
                potential_conditioning = list(set(neighbors_i + neighbors_j))
                
                # Test all conditioning sets of current size
                if len(potential_conditioning) >= conditioning_size:
                    for conditioning_vars in combinations(potential_conditioning, conditioning_size):
                        conditioning_indices = [variable_names.index(var) for var in conditioning_vars]
                        
                        if self._test_independence(X, i, j, conditioning_indices):
                            edges_to_remove.append((var_i, var_j))
                            break
            
            # Remove edges
            for edge in edges_to_remove:
                if graph.has_edge(*edge):
                    graph.remove_edge(*edge)
        
        # Convert to directed graph (simplified orientation rules)
        directed_graph = self._orient_edges(graph, X, variable_names)
        
        return directed_graph
    
    def _orient_edges(
        self,
        undirected_graph: nx.Graph,
        X: np.ndarray,
        variable_names: List[str]
    ) -> nx.DiGraph:
        """Orient edges using simplified rules"""
        
        # For simplicity, use correlation-based orientation
        # In practice, would use more sophisticated orientation rules
        
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(undirected_graph.nodes())
        
        for edge in undirected_graph.edges():
            var_i, var_j = edge
            i = variable_names.index(var_i)
            j = variable_names.index(var_j)
            
            # Simple heuristic: variable with higher variance causes the other
            # This is a placeholder - real orientation would use v-structures, etc.
            if np.var(X[:, i]) > np.var(X[:, j]):
                directed_graph.add_edge(var_i, var_j)
            else:
                directed_graph.add_edge(var_j, var_i)
        
        return directed_graph
    
    def estimate_causal_strength(self, X: np.ndarray, cause: int, effect: int) -> float:
        """Estimate causal strength using partial correlation"""
        
        # Find optimal conditioning set (simplified)
        n_vars = X.shape[1]
        other_vars = [i for i in range(n_vars) if i != cause and i != effect]
        
        # Try different conditioning sets and pick the one that maximizes partial correlation
        best_strength = 0
        
        for conditioning_size in range(min(len(other_vars), self.max_conditioning_set_size) + 1):
            if conditioning_size == 0:
                strength = abs(np.corrcoef(X[:, cause], X[:, effect])[0, 1])
                best_strength = max(best_strength, strength)
            else:
                for conditioning_vars in combinations(other_vars, conditioning_size):
                    try:
                        data_subset = X[:, [cause, effect] + list(conditioning_vars)]
                        partial_corr = self._partial_correlation(
                            data_subset, 0, 1, list(range(2, data_subset.shape[1]))
                        )
                        strength = abs(partial_corr)
                        best_strength = max(best_strength, strength)
                    except:
                        continue
        
        return best_strength


class GrangerCausality(CausalDiscoveryMethod):
    """
    Granger Causality for time series causal discovery
    """
    
    def __init__(
        self,
        max_lag: int = 5,
        alpha: float = 0.05,
        method: str = 'linear'  # 'linear', 'nonlinear'
    ):
        self.max_lag = max_lag
        self.alpha = alpha
        self.method = method
    
    def _granger_test(
        self,
        cause_series: np.ndarray,
        effect_series: np.ndarray,
        max_lag: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Test Granger causality between two time series
        
        Returns:
            f_statistic: F-statistic
            p_value: P-value
        """
        
        if max_lag is None:
            max_lag = self.max_lag
        
        n = len(effect_series)
        
        # Create lagged variables
        X_restricted = []  # Only lags of effect variable
        X_full = []       # Lags of both cause and effect variables
        y = []
        
        for t in range(max_lag, n):
            # Lagged effect variables
            effect_lags = [effect_series[t-i-1] for i in range(max_lag)]
            
            # Lagged cause variables
            cause_lags = [cause_series[t-i-1] for i in range(max_lag)]
            
            X_restricted.append(effect_lags)
            X_full.append(effect_lags + cause_lags)
            y.append(effect_series[t])
        
        X_restricted = np.array(X_restricted)
        X_full = np.array(X_full)
        y = np.array(y)
        
        # Fit restricted model (without cause)
        if self.method == 'linear':
            reg_restricted = LinearRegression().fit(X_restricted, y)
            rss_restricted = np.sum((y - reg_restricted.predict(X_restricted))**2)
            
            # Fit full model (with cause)
            reg_full = LinearRegression().fit(X_full, y)
            rss_full = np.sum((y - reg_full.predict(X_full))**2)
        
        elif self.method == 'nonlinear':
            # Use random forest for nonlinear relationships
            reg_restricted = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_restricted, y)
            rss_restricted = np.sum((y - reg_restricted.predict(X_restricted))**2)
            
            reg_full = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_full, y)
            rss_full = np.sum((y - reg_full.predict(X_full))**2)
        
        # F-test
        p = max_lag  # Number of additional parameters
        n_obs = len(y)
        k = X_full.shape[1]  # Total parameters in full model
        
        f_statistic = ((rss_restricted - rss_full) / p) / (rss_full / (n_obs - k - 1))
        p_value = 1 - stats.f.cdf(f_statistic, p, n_obs - k - 1)
        
        return f_statistic, p_value
    
    def discover_graph(self, X: np.ndarray, variable_names: Optional[List[str]] = None) -> nx.DiGraph:
        """Discover causal graph using Granger causality"""
        
        n_vars = X.shape[1]
        
        if variable_names is None:
            variable_names = [f"X{i}" for i in range(n_vars)]
        
        # Create directed graph
        graph = nx.DiGraph()
        graph.add_nodes_from(variable_names)
        
        # Test all pairs
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    f_stat, p_value = self._granger_test(X[:, i], X[:, j])
                    
                    if p_value < self.alpha:
                        # i Granger-causes j
                        graph.add_edge(variable_names[i], variable_names[j], 
                                     weight=f_stat, p_value=p_value)
        
        return graph
    
    def estimate_causal_strength(self, X: np.ndarray, cause: int, effect: int) -> float:
        """Estimate causal strength using F-statistic"""
        f_stat, _ = self._granger_test(X[:, cause], X[:, effect])
        return f_stat


class LiNGAM(CausalDiscoveryMethod):
    """
    Linear Non-Gaussian Acyclic Model (LiNGAM) for causal discovery
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.causal_order = None
        self.adjacency_matrix = None
    
    def _estimate_ica_mixing_matrix(self, X: np.ndarray) -> np.ndarray:
        """Estimate ICA mixing matrix"""
        try:
            from sklearn.decomposition import FastICA
            ica = FastICA(n_components=X.shape[1], random_state=42, max_iter=1000)
            ica.fit(X)
            return ica.mixing_
        except:
            # Fallback to simple approach
            return np.linalg.pinv(np.cov(X.T))
    
    def _find_causal_order(self, mixing_matrix: np.ndarray) -> List[int]:
        """Find causal ordering from mixing matrix"""
        
        n_vars = mixing_matrix.shape[0]
        W = mixing_matrix.copy()
        
        # Permutation matrix to make W lower triangular
        causal_order = []
        remaining_vars = list(range(n_vars))
        
        for _ in range(n_vars):
            # Find variable with smallest sum of absolute values in remaining rows
            min_sum = float('inf')
            best_var = 0
            
            for var in remaining_vars:
                row_sum = np.sum(np.abs(W[var, remaining_vars]))
                if row_sum < min_sum:
                    min_sum = row_sum
                    best_var = var
            
            causal_order.append(best_var)
            remaining_vars.remove(best_var)
        
        return causal_order
    
    def discover_graph(self, X: np.ndarray, variable_names: Optional[List[str]] = None) -> nx.DiGraph:
        """Discover causal graph using LiNGAM"""
        
        n_vars = X.shape[1]
        
        if variable_names is None:
            variable_names = [f"X{i}" for i in range(n_vars)]
        
        # Standardize data
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Estimate mixing matrix using ICA
        mixing_matrix = self._estimate_ica_mixing_matrix(X_std)
        
        # Find causal order
        self.causal_order = self._find_causal_order(mixing_matrix)
        
        # Estimate adjacency matrix
        self.adjacency_matrix = np.zeros((n_vars, n_vars))
        
        for i, cause_idx in enumerate(self.causal_order):
            for j, effect_idx in enumerate(self.causal_order[i+1:], i+1):
                # Regress effect on all previous causes
                causes = [self.causal_order[k] for k in range(j)]
                
                if len(causes) > 0:
                    X_causes = X_std[:, causes]
                    y_effect = X_std[:, effect_idx]
                    
                    reg = LinearRegression().fit(X_causes, y_effect)
                    
                    # Get coefficient for current cause
                    cause_position = causes.index(cause_idx)
                    coeff = reg.coef_[cause_position]
                    
                    # Test significance
                    residuals = y_effect - reg.predict(X_causes)
                    mse = np.mean(residuals**2)
                    
                    # Approximate standard error
                    X_var = np.var(X_causes[:, cause_position])
                    se = np.sqrt(mse / (len(X_std) * X_var))
                    
                    t_stat = coeff / se if se > 0 else 0
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(X_std) - len(causes) - 1))
                    
                    if p_value < self.alpha:
                        self.adjacency_matrix[cause_idx, effect_idx] = coeff
        
        # Create directed graph
        graph = nx.DiGraph()
        graph.add_nodes_from(variable_names)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if abs(self.adjacency_matrix[i, j]) > 1e-6:
                    graph.add_edge(variable_names[i], variable_names[j], 
                                 weight=self.adjacency_matrix[i, j])
        
        return graph
    
    def estimate_causal_strength(self, X: np.ndarray, cause: int, effect: int) -> float:
        """Estimate causal strength from adjacency matrix"""
        if self.adjacency_matrix is not None:
            return abs(self.adjacency_matrix[cause, effect])
        return 0.0


class CausalDiscoveryEnsemble:
    """
    Ensemble method combining multiple causal discovery approaches
    """
    
    def __init__(
        self,
        methods: Optional[List[CausalDiscoveryMethod]] = None,
        voting_threshold: float = 0.5
    ):
        if methods is None:
            methods = [
                PCAlgorithm(alpha=0.05),
                GrangerCausality(max_lag=3),
                LiNGAM(alpha=0.05)
            ]
        
        self.methods = methods
        self.voting_threshold = voting_threshold
    
    def discover_graph(
        self,
        X: np.ndarray,
        variable_names: Optional[List[str]] = None,
        time_series: bool = False
    ) -> nx.DiGraph:
        """
        Discover causal graph using ensemble voting
        
        Args:
            X: Data matrix
            variable_names: Variable names
            time_series: Whether data is time series (affects method selection)
        
        Returns:
            Consensus causal graph
        """
        
        n_vars = X.shape[1]
        
        if variable_names is None:
            variable_names = [f"X{i}" for i in range(n_vars)]
        
        # Get graphs from all methods
        graphs = []
        for method in self.methods:
            # Skip Granger causality for non-time series data
            if isinstance(method, GrangerCausality) and not time_series:
                continue
            
            try:
                graph = method.discover_graph(X, variable_names)
                graphs.append(graph)
            except Exception as e:
                warnings.warn(f"Method {type(method).__name__} failed: {e}")
                continue
        
        # Ensemble voting
        consensus_graph = nx.DiGraph()
        consensus_graph.add_nodes_from(variable_names)
        
        # Count votes for each edge
        edge_votes = {}
        
        for graph in graphs:
            for edge in graph.edges():
                if edge not in edge_votes:
                    edge_votes[edge] = 0
                edge_votes[edge] += 1
        
        # Add edges that meet voting threshold
        total_methods = len(graphs)
        
        for edge, votes in edge_votes.items():
            if votes / total_methods >= self.voting_threshold:
                # Compute average weight across methods
                weights = []
                for graph in graphs:
                    if graph.has_edge(*edge):
                        edge_data = graph.get_edge_data(*edge)
                        if 'weight' in edge_data:
                            weights.append(abs(edge_data['weight']))
                
                avg_weight = np.mean(weights) if weights else 1.0
                consensus_graph.add_edge(*edge, weight=avg_weight, votes=votes)
        
        return consensus_graph
    
    def estimate_causal_strength(
        self,
        X: np.ndarray,
        cause: int,
        effect: int,
        time_series: bool = False
    ) -> float:
        """Estimate causal strength using ensemble average"""
        
        strengths = []
        
        for method in self.methods:
            if isinstance(method, GrangerCausality) and not time_series:
                continue
            
            try:
                strength = method.estimate_causal_strength(X, cause, effect)
                strengths.append(strength)
            except:
                continue
        
        return np.mean(strengths) if strengths else 0.0


def evaluate_causal_discovery(
    true_graph: nx.DiGraph,
    discovered_graph: nx.DiGraph
) -> Dict[str, float]:
    """
    Evaluate causal discovery performance
    
    Args:
        true_graph: Ground truth causal graph
        discovered_graph: Discovered causal graph
    
    Returns:
        Performance metrics
    """
    
    # Get edge sets
    true_edges = set(true_graph.edges())
    discovered_edges = set(discovered_graph.edges())
    
    # Compute metrics
    true_positives = len(true_edges & discovered_edges)
    false_positives = len(discovered_edges - true_edges)
    false_negatives = len(true_edges - discovered_edges)
    
    # Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Structural Hamming Distance
    all_possible_edges = set(permutations(true_graph.nodes(), 2))
    shd = len((true_edges ^ discovered_edges) & all_possible_edges)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'structural_hamming_distance': shd,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    print("Testing Causal Discovery Methods...")
    
    # Generate synthetic causal data
    n_samples = 500
    
    # Create true causal structure: X0 -> X1 -> X2, X0 -> X2
    X0 = np.random.normal(0, 1, n_samples)
    X1 = 0.5 * X0 + np.random.normal(0, 0.5, n_samples)
    X2 = 0.3 * X0 + 0.7 * X1 + np.random.normal(0, 0.3, n_samples)
    
    X = np.column_stack([X0, X1, X2])
    variable_names = ['X0', 'X1', 'X2']
    
    # True causal graph
    true_graph = nx.DiGraph()
    true_graph.add_nodes_from(variable_names)
    true_graph.add_edges_from([('X0', 'X1'), ('X1', 'X2'), ('X0', 'X2')])
    
    print(f"True causal edges: {list(true_graph.edges())}")
    
    # Test different methods
    methods = {
        'PC Algorithm': PCAlgorithm(alpha=0.1),
        'LiNGAM': LiNGAM(alpha=0.1),
        'Ensemble': CausalDiscoveryEnsemble(voting_threshold=0.5)
    }
    
    for method_name, method in methods.items():
        print(f"\n{method_name} Results:")
        
        try:
            discovered_graph = method.discover_graph(X, variable_names)
            discovered_edges = list(discovered_graph.edges())
            
            print(f"  Discovered edges: {discovered_edges}")
            
            # Evaluate performance
            metrics = evaluate_causal_discovery(true_graph, discovered_graph)
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1_score']:.3f}")
            
            # Test causal strength estimation
            if hasattr(method, 'estimate_causal_strength'):
                strength_01 = method.estimate_causal_strength(X, 0, 1)
                strength_12 = method.estimate_causal_strength(X, 1, 2)
                print(f"  Causal strength X0->X1: {strength_01:.3f}")
                print(f"  Causal strength X1->X2: {strength_12:.3f}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test Granger causality with time series
    print("\nTesting Granger Causality with Time Series...")
    
    # Generate time series data
    n_time = 200
    ts1 = np.random.normal(0, 1, n_time)
    ts2 = np.zeros(n_time)
    
    # ts1 causes ts2 with lag
    for t in range(2, n_time):
        ts2[t] = 0.3 * ts1[t-1] + 0.2 * ts1[t-2] + np.random.normal(0, 0.5)
    
    ts_data = np.column_stack([ts1, ts2])
    
    granger = GrangerCausality(max_lag=3, alpha=0.05)
    ts_graph = granger.discover_graph(ts_data, ['TS1', 'TS2'])
    
    print(f"Time series causal edges: {list(ts_graph.edges())}")
    
    # Test causality in both directions
    f_stat_12, p_val_12 = granger._granger_test(ts1, ts2)
    f_stat_21, p_val_21 = granger._granger_test(ts2, ts1)
    
    print(f"TS1 -> TS2: F-stat={f_stat_12:.3f}, p-value={p_val_12:.3f}")
    print(f"TS2 -> TS1: F-stat={f_stat_21:.3f}, p-value={p_val_21:.3f}")
    
    print("\nDone!")