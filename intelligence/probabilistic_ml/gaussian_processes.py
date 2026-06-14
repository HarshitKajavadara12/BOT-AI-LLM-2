from __future__ import annotations

"""
Gaussian Processes for Financial Modeling
Advanced GP implementations for uncertainty quantification and prediction
"""

import torch
import torch.nn as nn
try:
    import gpytorch
    from gpytorch.models import ExactGP, ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
    from gpytorch.means import ConstantMean, LinearMean
    from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, SpectralMixureKernel
    from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
    from gpytorch.distributions import MultivariateNormal
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    gpytorch = None
    # Stub base classes so the module can be imported without gpytorch
    class ExactGP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    class ApproximateGP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    ConstantMean = LinearMean = None
    ScaleKernel = RBFKernel = MaternKernel = SpectralMixureKernel = None
    GaussianLikelihood = BernoulliLikelihood = None
    MultivariateNormal = None
    CholeskyVariationalDistribution = VariationalStrategy = None
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


class FinancialGaussianProcess(ExactGP):
    """
    Gaussian Process model specifically designed for financial time series
    """
    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel_type: str = 'rbf',
        mean_type: str = 'constant',
        ard_num_dims: Optional[int] = None
    ):
        super().__init__(train_x, train_y, likelihood)
        
        # Mean function
        if mean_type == 'constant':
            self.mean_module = ConstantMean()
        elif mean_type == 'linear':
            self.mean_module = LinearMean(train_x.size(-1))
        else:
            raise ValueError(f"Unknown mean type: {mean_type}")
        
        # Kernel function
        base_kernel = self._create_kernel(kernel_type, train_x.size(-1), ard_num_dims)
        self.covar_module = ScaleKernel(base_kernel)
        
    def _create_kernel(
        self,
        kernel_type: str,
        input_dim: int,
        ard_num_dims: Optional[int]
    ) -> gpytorch.kernels.Kernel:
        """Create kernel based on type"""
        
        if kernel_type == 'rbf':
            return RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern':
            return MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        elif kernel_type == 'spectral_mixture':
            return SpectralMixureKernel(num_mixtures=4, ard_num_dims=ard_num_dims)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SparseGaussianProcess(ApproximateGP):
    """
    Sparse Gaussian Process for large-scale financial data
    """
    
    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel_type: str = 'rbf',
        num_inducing: int = 100
    ):
        variational_distribution = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution
        )
        super().__init__(variational_strategy)
        
        self.mean_module = ConstantMean()
        
        # Kernel
        if kernel_type == 'rbf':
            base_kernel = RBFKernel()
        elif kernel_type == 'matern':
            base_kernel = MaternKernel(nu=2.5)
        else:
            base_kernel = RBFKernel()
            
        self.covar_module = ScaleKernel(base_kernel)
        
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultiTaskGaussianProcess(ExactGP):
    """
    Multi-task Gaussian Process for modeling multiple assets simultaneously
    """
    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        num_tasks: int
    ):
        super().__init__(train_x, train_y, likelihood)
        
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            ConstantMean(), num_tasks=num_tasks
        )
        
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            RBFKernel(), num_tasks=num_tasks, rank=1
        )
        
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class VolatilityGaussianProcess(ExactGP):
    """
    Specialized GP for volatility modeling with heteroskedastic noise
    """
    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood
    ):
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = ConstantMean()
        
        # Use Matern kernel for volatility (captures rough paths)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, ard_num_dims=train_x.size(-1))
        )
        
        # Add periodic component for intraday patterns
        self.periodic_module = gpytorch.kernels.PeriodicKernel()
        
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) + self.periodic_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepGaussianProcess(nn.Module):
    """
    Deep Gaussian Process with multiple layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_inducing: int = 100,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dims = [input_dim] + hidden_dims
        
        # Create GP layers
        self.gp_layers = nn.ModuleList()
        
        for i in range(num_layers):
            input_dim_i = self.hidden_dims[i]
            output_dim_i = self.hidden_dims[i + 1] if i < num_layers - 1 else 1
            
            # Inducing points for this layer
            inducing_points = torch.randn(num_inducing, input_dim_i)
            
            gp_layer = SparseGaussianProcess(inducing_points)
            self.gp_layers.append(gp_layer)
        
        self.likelihood = GaussianLikelihood()
        
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through deep GP"""
        current_x = x
        
        for i, gp_layer in enumerate(self.gp_layers):
            if i == len(self.gp_layers) - 1:
                # Final layer
                return gp_layer(current_x)
            else:
                # Intermediate layer - sample from GP
                gp_output = gp_layer(current_x)
                current_x = gp_output.sample()
        
        return gp_output


class GPRegimeModel:
    """
    Gaussian Process model that switches between different market regimes
    """
    
    def __init__(
        self,
        num_regimes: int = 3,
        regime_features: Optional[List[str]] = None
    ):
        self.num_regimes = num_regimes
        self.regime_features = regime_features or ['volatility', 'volume', 'trend']
        
        # GP models for each regime
        self.regime_gps = {}
        self.regime_likelihoods = {}
        
        # Regime classifier (simple threshold-based for now)
        self.regime_thresholds = None
        
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        regime_indicators: Optional[torch.Tensor] = None
    ):
        """
        Fit GP models for each regime
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            regime_indicators: Optional regime labels [n_samples]
        """
        
        if regime_indicators is None:
            # Automatically detect regimes using clustering
            regime_indicators = self._detect_regimes(X, y)
        
        # Fit GP for each regime
        for regime in range(self.num_regimes):
            regime_mask = regime_indicators == regime
            
            if regime_mask.sum() > 10:  # Minimum samples per regime
                X_regime = X[regime_mask]
                y_regime = y[regime_mask]
                
                likelihood = GaussianLikelihood()
                gp_model = FinancialGaussianProcess(
                    X_regime, y_regime, likelihood, kernel_type='matern'
                )
                
                self.regime_gps[regime] = gp_model
                self.regime_likelihoods[regime] = likelihood
    
    def _detect_regimes(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Detect market regimes using volatility clustering"""
        from sklearn.cluster import KMeans
        
        # Use rolling volatility as regime indicator
        returns = torch.diff(y)
        rolling_vol = torch.zeros_like(y)
        
        window = 20
        for i in range(window, len(y)):
            rolling_vol[i] = torch.std(returns[i-window:i])
        
        # Cluster based on volatility
        vol_features = rolling_vol.numpy().reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.num_regimes, random_state=42)
        regimes = kmeans.fit_predict(vol_features)
        
        return torch.tensor(regimes)
    
    def predict(
        self,
        X: torch.Tensor,
        return_std: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions by mixing regime-specific GPs
        
        Args:
            X: Input features [n_samples, n_features]
            return_std: Whether to return prediction standard deviation
        
        Returns:
            predictions: Mean predictions [n_samples]
            std: Standard deviations [n_samples] (if return_std=True)
        """
        
        # Classify regime for each sample
        regime_probs = self._classify_regimes(X)
        
        predictions = torch.zeros(X.shape[0])
        variances = torch.zeros(X.shape[0])
        
        for i, x_i in enumerate(X):
            pred_i = 0
            var_i = 0
            
            for regime in range(self.num_regimes):
                if regime in self.regime_gps:
                    gp = self.regime_gps[regime]
                    likelihood = self.regime_likelihoods[regime]
                    
                    gp.eval()
                    likelihood.eval()
                    
                    with torch.no_grad():
                        posterior = likelihood(gp(x_i.unsqueeze(0)))
                        regime_pred = posterior.mean
                        regime_var = posterior.variance
                    
                    # Weight by regime probability
                    weight = regime_probs[i, regime]
                    pred_i += weight * regime_pred
                    var_i += weight * regime_var
            
            predictions[i] = pred_i
            variances[i] = var_i
        
        if return_std:
            return predictions, torch.sqrt(variances)
        return predictions
    
    def _classify_regimes(self, X: torch.Tensor) -> torch.Tensor:
        """Classify samples into regimes"""
        # Simple volatility-based classification
        # In practice, this would use a more sophisticated classifier
        
        regime_probs = torch.zeros(X.shape[0], self.num_regimes)
        
        # Use first feature as volatility proxy
        vol_proxy = X[:, 0] if X.shape[1] > 0 else torch.randn(X.shape[0])
        
        # Simple threshold-based classification
        low_vol_thresh = torch.quantile(vol_proxy, 0.33)
        high_vol_thresh = torch.quantile(vol_proxy, 0.67)
        
        for i, vol in enumerate(vol_proxy):
            if vol < low_vol_thresh:
                regime_probs[i, 0] = 1.0  # Low volatility regime
            elif vol > high_vol_thresh:
                regime_probs[i, 2] = 1.0  # High volatility regime
            else:
                regime_probs[i, 1] = 1.0  # Medium volatility regime
        
        return regime_probs


class OnlineGaussianProcess:
    """
    Online GP that can update incrementally with new data
    """
    
    def __init__(
        self,
        initial_x: torch.Tensor,
        initial_y: torch.Tensor,
        kernel_type: str = 'rbf',
        max_inducing: int = 500,
        forgetting_factor: float = 0.99
    ):
        self.max_inducing = max_inducing
        self.forgetting_factor = forgetting_factor
        
        # Initialize with sparse GP
        self.likelihood = GaussianLikelihood()
        
        # Start with initial inducing points
        num_initial = min(len(initial_x), max_inducing)
        inducing_indices = torch.randperm(len(initial_x))[:num_initial]
        inducing_points = initial_x[inducing_indices]
        
        self.model = SparseGaussianProcess(inducing_points, kernel_type)
        
        # Training data
        self.train_x = initial_x.clone()
        self.train_y = initial_y.clone()
        
        # Fit initial model
        self._fit_model()
        
    def _fit_model(self):
        """Fit the GP model"""
        self.model.train()
        self.likelihood.train()
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=0.01
        )
        
        # Training loop
        num_iter = 100
        for i in range(num_iter):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -gpytorch.mlls.VariationalELBO(
                self.likelihood, self.model, num_data=self.train_y.size(0)
            )(output, self.train_y)
            loss.backward()
            optimizer.step()
    
    def update(self, new_x: torch.Tensor, new_y: torch.Tensor):
        """
        Update GP with new data point
        
        Args:
            new_x: New input [1, n_features] or [n_new, n_features]
            new_y: New target [1] or [n_new]
        """
        
        # Add new data
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])
        
        # Apply forgetting factor to old data
        n_old = len(self.train_y) - len(new_y)
        weights = torch.cat([
            torch.full((n_old,), self.forgetting_factor),
            torch.ones(len(new_y))
        ])
        
        # Subsample if too many points
        if len(self.train_x) > self.max_inducing * 2:
            # Keep most recent and high-weight points
            keep_indices = torch.argsort(weights)[-self.max_inducing:]
            self.train_x = self.train_x[keep_indices]
            self.train_y = self.train_y[keep_indices]
        
        # Update inducing points if needed
        if len(self.train_x) > self.max_inducing:
            # Select new inducing points
            inducing_indices = torch.randperm(len(self.train_x))[:self.max_inducing//2]
            new_inducing = self.train_x[inducing_indices]
            
            # Update model
            self.model.variational_strategy.inducing_points.data = new_inducing
        
        # Refit model
        self._fit_model()
    
    def predict(
        self,
        test_x: torch.Tensor,
        return_std: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Make predictions"""
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            posterior = self.likelihood(self.model(test_x))
            mean = posterior.mean
            
            if return_std:
                std = torch.sqrt(posterior.variance)
                return mean, std
            return mean, None


def create_gp_model(
    model_type: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    **kwargs
) -> nn.Module:
    """
    Factory function to create GP models
    
    Args:
        model_type: Type of GP model
        train_x: Training inputs
        train_y: Training targets
        **kwargs: Additional parameters
    
    Returns:
        Configured GP model
    """
    
    likelihood = GaussianLikelihood()
    
    if model_type == 'exact':
        return FinancialGaussianProcess(train_x, train_y, likelihood, **kwargs)
    elif model_type == 'sparse':
        inducing_points = train_x[torch.randperm(len(train_x))[:kwargs.get('num_inducing', 100)]]
        return SparseGaussianProcess(inducing_points, **kwargs)
    elif model_type == 'multitask':
        return MultiTaskGaussianProcess(train_x, train_y, likelihood, **kwargs)
    elif model_type == 'volatility':
        return VolatilityGaussianProcess(train_x, train_y, likelihood)
    elif model_type == 'deep':
        return DeepGaussianProcess(train_x.size(-1), **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    
    # Generate synthetic financial data
    n_samples = 1000
    n_features = 5
    
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + 
         torch.sin(X[:, 2]) + torch.randn(n_samples) * 0.1)
    
    # Split data
    train_size = int(0.8 * n_samples)
    train_x, test_x = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    
    # Test different GP models
    models = {
        'exact': create_gp_model('exact', train_x, train_y, kernel_type='rbf'),
        'sparse': create_gp_model('sparse', train_x, train_y, num_inducing=100),
        'volatility': create_gp_model('volatility', train_x, train_y)
    }
    
    for name, model in models.items():
        print(f"Testing {name} GP model...")
        
        # Train model
        model.train()
        likelihood = GaussianLikelihood()
        likelihood.train()
        
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()), lr=0.1
        )
        
        # Training loop (simplified)
        for i in range(50):
            optimizer.zero_grad()
            output = model(train_x)
            
            if isinstance(model, SparseGaussianProcess):
                loss = -gpytorch.mlls.VariationalELBO(
                    likelihood, model, num_data=train_y.size(0)
                )(output, train_y)
            else:
                loss = -gpytorch.mlls.ExactMarginalLogLikelihood(
                    likelihood, model
                )(output, train_y)
            
            loss.backward()
            optimizer.step()
        
        # Test prediction
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            pred = likelihood(model(test_x))
            mean = pred.mean
            std = torch.sqrt(pred.variance)
            
            # Compute RMSE
            rmse = torch.sqrt(torch.mean((mean - test_y)**2))
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Mean prediction std: {std.mean():.4f}")
    
    print("\nTesting regime-switching GP...")
    regime_gp = GPRegimeModel(num_regimes=2)
    regime_gp.fit(train_x, train_y)
    
    pred_mean, pred_std = regime_gp.predict(test_x)
    rmse = torch.sqrt(torch.mean((pred_mean - test_y)**2))
    print(f"  Regime GP RMSE: {rmse:.4f}")
    
    print("\nTesting online GP...")
    online_gp = OnlineGaussianProcess(train_x[:100], train_y[:100])
    
    # Simulate online updates
    for i in range(100, len(train_x), 10):
        end_idx = min(i + 10, len(train_x))
        online_gp.update(train_x[i:end_idx], train_y[i:end_idx])
    
    pred_mean, pred_std = online_gp.predict(test_x)
    rmse = torch.sqrt(torch.mean((pred_mean - test_y)**2))
    print(f"  Online GP RMSE: {rmse:.4f}")