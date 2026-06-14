"""
Bayesian Optimization for Financial Strategy Optimization
Advanced BO implementations for hyperparameter tuning and strategy optimization
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod

# Import BoTorch components
try:
    import botorch
    from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
    from botorch.fit import fit_gpytorch_model
    from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, qExpectedImprovement
    from botorch.acquisition.monte_carlo import qUpperConfidenceBound
    from botorch.optim import optimize_acqf
    from botorch.utils.transforms import unnormalize, normalize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.means import ConstantMean
    from gpytorch.priors import GammaPrior
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    warnings.warn("BoTorch not available. Some functionality will be limited.")


@dataclass
class OptimizationResult:
    """Results from Bayesian optimization"""
    best_params: Dict[str, float]
    best_value: float
    optimization_history: List[Tuple[Dict[str, float], float]]
    model: Optional[Any] = None
    acquisition_values: Optional[List[float]] = None


class ObjectiveFunction(ABC):
    """Abstract base class for optimization objectives"""
    
    @abstractmethod
    def __call__(self, params: Dict[str, float]) -> float:
        """Evaluate objective function"""
        pass
    
    @abstractmethod
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds"""
        pass


class PortfolioObjective(ObjectiveFunction):
    """
    Portfolio optimization objective function
    """
    
    def __init__(
        self,
        returns_data: np.ndarray,
        risk_free_rate: float = 0.02,
        lookback_window: int = 252,
        transaction_costs: float = 0.001
    ):
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        self.lookback_window = lookback_window
        self.transaction_costs = transaction_costs
        
    def __call__(self, params: Dict[str, float]) -> float:
        """
        Evaluate portfolio performance given parameters
        
        Args:
            params: Portfolio parameters (weights, rebalancing frequency, etc.)
        
        Returns:
            Negative Sharpe ratio (for minimization)
        """
        
        # Extract parameters
        weights = np.array([params.get(f'weight_{i}', 0) for i in range(self.returns_data.shape[1])])
        rebalance_freq = int(params.get('rebalance_freq', 21))
        
        # Normalize weights
        weights = weights / np.sum(np.abs(weights))
        
        # Compute portfolio returns
        portfolio_returns = self._compute_portfolio_returns(weights, rebalance_freq)
        
        # Compute Sharpe ratio
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Return negative Sharpe ratio (for minimization)
        return -sharpe_ratio
    
    def _compute_portfolio_returns(self, weights: np.ndarray, rebalance_freq: int) -> np.ndarray:
        """Compute portfolio returns with rebalancing"""
        n_periods = len(self.returns_data)
        portfolio_returns = np.zeros(n_periods)
        current_weights = weights.copy()
        
        for t in range(n_periods):
            # Compute return for this period
            portfolio_returns[t] = np.sum(current_weights * self.returns_data[t])
            
            # Update weights due to price changes
            current_weights *= (1 + self.returns_data[t])
            current_weights /= np.sum(current_weights)
            
            # Rebalance if needed
            if t % rebalance_freq == 0 and t > 0:
                rebalancing_cost = np.sum(np.abs(current_weights - weights)) * self.transaction_costs
                portfolio_returns[t] -= rebalancing_cost
                current_weights = weights.copy()
        
        return portfolio_returns
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds"""
        bounds = {}
        n_assets = self.returns_data.shape[1]
        
        # Weight bounds
        for i in range(n_assets):
            bounds[f'weight_{i}'] = (-0.3, 0.3)  # Max 30% position
        
        # Rebalancing frequency bounds
        bounds['rebalance_freq'] = (1, 63)  # Daily to quarterly
        
        return bounds


class HyperparameterObjective(ObjectiveFunction):
    """
    Hyperparameter optimization for ML models
    """
    
    def __init__(
        self,
        model_class: Callable,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        metric: str = 'mse'
    ):
        self.model_class = model_class
        self.train_x, self.train_y = train_data
        self.val_x, self.val_y = val_data
        self.metric = metric
        
    def __call__(self, params: Dict[str, float]) -> float:
        """
        Evaluate model performance with given hyperparameters
        
        Args:
            params: Model hyperparameters
        
        Returns:
            Validation loss
        """
        
        try:
            # Create model with hyperparameters
            model = self.model_class(**params)
            
            # Train model (simplified training loop)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
            
            model.train()
            for epoch in range(100):  # Fixed number of epochs
                optimizer.zero_grad()
                pred = model(self.train_x)
                loss = torch.nn.MSELoss()(pred, self.train_y)
                loss.backward()
                optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_pred = model(self.val_x)
                
                if self.metric == 'mse':
                    val_loss = torch.nn.MSELoss()(val_pred, self.val_y).item()
                elif self.metric == 'mae':
                    val_loss = torch.nn.L1Loss()(val_pred, self.val_y).item()
                else:
                    val_loss = torch.nn.MSELoss()(val_pred, self.val_y).item()
            
            return val_loss
            
        except Exception as e:
            # Return high loss if model fails
            return 1e6
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get hyperparameter bounds (must be implemented by subclass)"""
        return {
            'lr': (1e-5, 1e-1),
            'hidden_dim': (32, 512),
            'num_layers': (1, 5),
            'dropout': (0.0, 0.5)
        }


class BayesianOptimizer:
    """
    Bayesian Optimizer using Gaussian Processes
    """
    
    def __init__(
        self,
        objective_function: ObjectiveFunction,
        acquisition_function: str = 'ei',
        n_initial_points: int = 10,
        noise_level: float = 1e-6,
        normalize_y: bool = True,
        random_state: Optional[int] = None
    ):
        
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for BayesianOptimizer")
        
        self.objective_function = objective_function
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.noise_level = noise_level
        self.normalize_y = normalize_y
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # Get parameter bounds
        self.bounds_dict = objective_function.get_bounds()
        self.param_names = list(self.bounds_dict.keys())
        self.bounds = torch.tensor([
            [self.bounds_dict[name][0] for name in self.param_names],
            [self.bounds_dict[name][1] for name in self.param_names]
        ], dtype=torch.float64)
        
        # Initialize storage
        self.X = torch.empty(0, len(self.param_names), dtype=torch.float64)
        self.Y = torch.empty(0, 1, dtype=torch.float64)
        self.model = None
        
    def _params_dict_to_tensor(self, params_dict: Dict[str, float]) -> torch.Tensor:
        """Convert parameter dictionary to tensor"""
        return torch.tensor([params_dict[name] for name in self.param_names], dtype=torch.float64)
    
    def _tensor_to_params_dict(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Convert tensor to parameter dictionary"""
        return {name: tensor[i].item() for i, name in enumerate(self.param_names)}
    
    def _generate_initial_points(self) -> torch.Tensor:
        """Generate initial points using Latin Hypercube Sampling"""
        from scipy.stats import qmc
        
        # Use Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=len(self.param_names), seed=42)
        unit_samples = sampler.random(n=self.n_initial_points)
        
        # Scale to bounds
        lower_bounds = self.bounds[0].numpy()
        upper_bounds = self.bounds[1].numpy()
        
        scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)
        
        return torch.tensor(scaled_samples, dtype=torch.float64)
    
    def _fit_gp_model(self):
        """Fit Gaussian Process model to current data"""
        
        if len(self.X) == 0:
            return
        
        # Normalize inputs
        X_normalized = normalize(self.X, self.bounds)
        
        # Normalize outputs if requested
        if self.normalize_y:
            Y_mean = self.Y.mean()
            Y_std = self.Y.std()
            Y_normalized = (self.Y - Y_mean) / Y_std
        else:
            Y_normalized = self.Y
        
        # Create GP model
        if self.noise_level > 0:
            noise = torch.full_like(Y_normalized.squeeze(), self.noise_level)
            self.model = FixedNoiseGP(X_normalized, Y_normalized, noise.unsqueeze(-1))
        else:
            self.model = SingleTaskGP(X_normalized, Y_normalized)
        
        # Fit model
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)
        
    def _optimize_acquisition(self) -> torch.Tensor:
        """Optimize acquisition function to find next point"""
        
        if self.model is None:
            raise ValueError("Model must be fitted before optimizing acquisition function")
        
        # Create acquisition function
        if self.acquisition_function == 'ei':
            best_f = self.Y.max() if len(self.Y) > 0 else 0.0
            acq_func = ExpectedImprovement(self.model, best_f=best_f)
        elif self.acquisition_function == 'ucb':
            acq_func = UpperConfidenceBound(self.model, beta=2.0)
        elif self.acquisition_function == 'qei':
            best_f = self.Y.max() if len(self.Y) > 0 else 0.0
            acq_func = qExpectedImprovement(self.model, best_f=best_f)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
        
        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(len(self.param_names)), torch.ones(len(self.param_names))]),
            q=1,
            num_restarts=20,
            raw_samples=512,
        )
        
        # Unnormalize candidates
        next_point = unnormalize(candidates, self.bounds)
        
        return next_point.squeeze(0)
    
    def suggest_next_point(self) -> Dict[str, float]:
        """Suggest next point to evaluate"""
        
        if len(self.X) < self.n_initial_points:
            # Generate initial points
            if len(self.X) == 0:
                self.initial_points = self._generate_initial_points()
            
            next_point = self.initial_points[len(self.X)]
        else:
            # Use acquisition function
            self._fit_gp_model()
            next_point = self._optimize_acquisition()
        
        return self._tensor_to_params_dict(next_point)
    
    def update(self, params: Dict[str, float], value: float):
        """Update optimizer with new evaluation"""
        
        # Convert to tensor
        x_tensor = self._params_dict_to_tensor(params).unsqueeze(0)
        y_tensor = torch.tensor([[value]], dtype=torch.float64)
        
        # Add to dataset
        self.X = torch.cat([self.X, x_tensor])
        self.Y = torch.cat([self.Y, y_tensor])
    
    def optimize(self, n_iterations: int = 50) -> OptimizationResult:
        """
        Run Bayesian optimization
        
        Args:
            n_iterations: Number of optimization iterations
        
        Returns:
            Optimization results
        """
        
        history = []
        
        for i in range(n_iterations):
            # Suggest next point
            next_params = self.suggest_next_point()
            
            # Evaluate objective function
            value = self.objective_function(next_params)
            
            # Update optimizer
            self.update(next_params, value)
            
            # Store history
            history.append((next_params.copy(), value))
            
            print(f"Iteration {i+1}/{n_iterations}: Value = {value:.6f}")
        
        # Find best parameters
        best_idx = torch.argmin(self.Y).item()
        best_params = self._tensor_to_params_dict(self.X[best_idx])
        best_value = self.Y[best_idx].item()
        
        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            optimization_history=history,
            model=self.model
        )


class MultiObjectiveBayesianOptimizer:
    """
    Multi-objective Bayesian optimization using Pareto efficiency
    """
    
    def __init__(
        self,
        objective_functions: List[ObjectiveFunction],
        reference_point: Optional[List[float]] = None,
        n_initial_points: int = 20
    ):
        
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for MultiObjectiveBayesianOptimizer")
        
        self.objective_functions = objective_functions
        self.n_objectives = len(objective_functions)
        self.reference_point = reference_point
        self.n_initial_points = n_initial_points
        
        # Get parameter bounds (assume all objectives have same parameters)
        self.bounds_dict = objective_functions[0].get_bounds()
        self.param_names = list(self.bounds_dict.keys())
        self.bounds = torch.tensor([
            [self.bounds_dict[name][0] for name in self.param_names],
            [self.bounds_dict[name][1] for name in self.param_names]
        ], dtype=torch.float64)
        
        # Initialize storage
        self.X = torch.empty(0, len(self.param_names), dtype=torch.float64)
        self.Y = torch.empty(0, self.n_objectives, dtype=torch.float64)
        
    def _is_pareto_efficient(self, costs: torch.Tensor) -> torch.Tensor:
        """Find Pareto efficient points"""
        is_efficient = torch.zeros(costs.shape[0], dtype=torch.bool)
        
        for i, cost in enumerate(costs):
            # A point is Pareto efficient if no other point dominates it
            dominated = torch.all(costs <= cost, dim=1) & torch.any(costs < cost, dim=1)
            if not torch.any(dominated):
                is_efficient[i] = True
        
        return is_efficient
    
    def optimize(self, n_iterations: int = 100) -> Dict[str, Any]:
        """
        Run multi-objective Bayesian optimization
        
        Args:
            n_iterations: Number of optimization iterations
        
        Returns:
            Optimization results including Pareto front
        """
        
        # Generate initial points
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=len(self.param_names), seed=42)
        unit_samples = sampler.random(n=self.n_initial_points)
        
        lower_bounds = self.bounds[0].numpy()
        upper_bounds = self.bounds[1].numpy()
        scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)
        initial_points = torch.tensor(scaled_samples, dtype=torch.float64)
        
        # Evaluate initial points
        for point in initial_points:
            params_dict = {name: point[i].item() for i, name in enumerate(self.param_names)}
            values = [obj_func(params_dict) for obj_func in self.objective_functions]
            
            self.X = torch.cat([self.X, point.unsqueeze(0)])
            self.Y = torch.cat([self.Y, torch.tensor([values], dtype=torch.float64)])
        
        history = []
        
        for i in range(n_iterations - self.n_initial_points):
            # Fit multi-task GP
            model = ModelListGP(*[SingleTaskGP(self.X, self.Y[:, j:j+1]) for j in range(self.n_objectives)])
            
            # Find Pareto front
            pareto_mask = self._is_pareto_efficient(self.Y)
            pareto_Y = self.Y[pareto_mask]
            
            # Use hypervolume-based acquisition (simplified)
            # In practice, would use more sophisticated multi-objective acquisition
            if self.reference_point is None:
                ref_point = self.Y.max(dim=0)[0] + 0.1 * (self.Y.max(dim=0)[0] - self.Y.min(dim=0)[0])
            else:
                ref_point = torch.tensor(self.reference_point, dtype=torch.float64)
            
            # Generate candidate point (simplified random sampling)
            candidate = torch.rand(1, len(self.param_names), dtype=torch.float64)
            candidate = candidate * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            
            # Evaluate candidate
            params_dict = {name: candidate[0, i].item() for i, name in enumerate(self.param_names)}
            values = [obj_func(params_dict) for obj_func in self.objective_functions]
            
            self.X = torch.cat([self.X, candidate])
            self.Y = torch.cat([self.Y, torch.tensor([values], dtype=torch.float64)])
            
            history.append((params_dict.copy(), values))
            
            print(f"Iteration {i+1}/{n_iterations-self.n_initial_points}: Values = {values}")
        
        # Find final Pareto front
        pareto_mask = self._is_pareto_efficient(self.Y)
        pareto_X = self.X[pareto_mask]
        pareto_Y = self.Y[pareto_mask]
        
        pareto_solutions = []
        for i in range(len(pareto_X)):
            params = {name: pareto_X[i, j].item() for j, name in enumerate(self.param_names)}
            objectives = pareto_Y[i].tolist()
            pareto_solutions.append((params, objectives))
        
        return {
            'pareto_front': pareto_solutions,
            'all_evaluations': history,
            'pareto_objectives': pareto_Y,
            'pareto_parameters': pareto_X
        }


# Example objective functions for testing
class QuadraticObjective(ObjectiveFunction):
    """Simple quadratic objective for testing"""
    
    def __init__(self, dim: int = 2):
        self.dim = dim
    
    def __call__(self, params: Dict[str, float]) -> float:
        x = np.array([params[f'x{i}'] for i in range(self.dim)])
        return np.sum((x - 0.5)**2)
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {f'x{i}': (-2.0, 2.0) for i in range(self.dim)}


def optimize_trading_strategy(
    returns_data: np.ndarray,
    n_iterations: int = 50,
    acquisition_function: str = 'ei'
) -> OptimizationResult:
    """
    Optimize trading strategy using Bayesian optimization
    
    Args:
        returns_data: Historical returns data [n_days, n_assets]
        n_iterations: Number of optimization iterations
        acquisition_function: Acquisition function to use
    
    Returns:
        Optimization results
    """
    
    # Create portfolio objective
    objective = PortfolioObjective(returns_data)
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        objective_function=objective,
        acquisition_function=acquisition_function,
        n_initial_points=10
    )
    
    # Run optimization
    result = optimizer.optimize(n_iterations)
    
    return result


if __name__ == "__main__":
    print("Testing Bayesian Optimization...")
    
    # Test with simple quadratic function
    objective = QuadraticObjective(dim=3)
    optimizer = BayesianOptimizer(objective, n_initial_points=5)
    
    result = optimizer.optimize(n_iterations=20)
    
    print(f"Best parameters: {result.best_params}")
    print(f"Best value: {result.best_value:.6f}")
    print(f"True optimum should be near: {{'x0': 0.5, 'x1': 0.5, 'x2': 0.5}}")
    
    # Test portfolio optimization with deterministic synthetic data
    print("\nTesting Portfolio Optimization...")
    n_days, n_assets = 252, 5
    # Deterministic seasonal base returns (small sinusoidal seasonal effect)
    t = np.arange(n_days)
    base = 0.001 + 0.0005 * np.sin(2 * np.pi * t / 63)
    # Construct deterministic multi-asset returns by adding small asset-specific offsets
    returns_data = np.vstack([base + (i - (n_assets // 2)) * 0.0001 for i in range(n_assets)]).T
    # Add deterministic correlation-like effect: make asset 1 partially follow asset 0
    returns_data[:, 1] += 0.3 * returns_data[:, 0]
    
    try:
        portfolio_result = optimize_trading_strategy(returns_data, n_iterations=10)
        print(f"Best portfolio parameters: {portfolio_result.best_params}")
        print(f"Best Sharpe ratio: {-portfolio_result.best_value:.4f}")
    except Exception as e:
        print(f"Portfolio optimization failed: {e}")
    
    print("\nDone!")