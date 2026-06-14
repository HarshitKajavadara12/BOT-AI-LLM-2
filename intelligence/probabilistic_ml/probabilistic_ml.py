"""
Probabilistic Machine Learning Models for QUANTUM-FORGE
Implements Bayesian networks, Gaussian processes, variational inference, and uncertainty quantification.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal, Categorical, Dirichlet
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import warnings
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular
import pickle
warnings.filterwarnings('ignore')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProbabilisticModelType(Enum):
    """Types of probabilistic models."""
    BAYESIAN_NETWORK = "bayesian_network"
    GAUSSIAN_PROCESS = "gaussian_process"
    VARIATIONAL_AUTOENCODER = "variational_autoencoder"
    BAYESIAN_NEURAL_NETWORK = "bayesian_neural_network"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    DIRICHLET_PROCESS = "dirichlet_process"
    HIDDEN_MARKOV = "hidden_markov"

class KernelType(Enum):
    """Kernel types for Gaussian processes."""
    RBF = "rbf"
    MATERN = "matern"
    PERIODIC = "periodic"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    WHITE_NOISE = "white_noise"

@dataclass
class ProbabilisticConfig:
    """Configuration for probabilistic models."""
    model_type: ProbabilisticModelType
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_components: int  # For mixture models
    learning_rate: float
    num_epochs: int
    batch_size: int
    prior_std: float
    likelihood_std: float
    kl_weight: float  # For VAE
    mc_samples: int  # Monte Carlo samples

class GaussianProcess:
    """Gaussian Process regression with various kernels."""
    
    def __init__(self, kernel_type: KernelType = KernelType.RBF, 
                 length_scale: float = 1.0, variance: float = 1.0,
                 noise_variance: float = 1e-6):
        """Initialize Gaussian Process."""
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.variance = variance
        self.noise_variance = noise_variance
        
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        
    def kernel_function(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix between two sets of points."""
        if self.kernel_type == KernelType.RBF:
            # RBF (Squared Exponential) kernel
            dists = torch.cdist(X1, X2, p=2) ** 2
            return self.variance * torch.exp(-0.5 * dists / (self.length_scale ** 2))
            
        elif self.kernel_type == KernelType.MATERN:
            # Matérn 5/2 kernel
            dists = torch.cdist(X1, X2, p=2)
            sqrt5_r = np.sqrt(5) * dists / self.length_scale
            return self.variance * (1 + sqrt5_r + (5/3) * (dists ** 2) / (self.length_scale ** 2)) * \
                   torch.exp(-sqrt5_r)
                   
        elif self.kernel_type == KernelType.PERIODIC:
            # Periodic kernel
            dists = torch.cdist(X1, X2, p=2)
            return self.variance * torch.exp(-2 * torch.sin(np.pi * dists / self.length_scale) ** 2)
            
        elif self.kernel_type == KernelType.LINEAR:
            # Linear kernel
            return self.variance * torch.matmul(X1, X2.T)
            
        elif self.kernel_type == KernelType.POLYNOMIAL:
            # Polynomial kernel (degree 2)
            return self.variance * (torch.matmul(X1, X2.T) + 1) ** 2
            
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Fit Gaussian Process to training data."""
        self.X_train = X
        self.y_train = y
        
        # Compute kernel matrix
        K = self.kernel_function(X, X)
        K += self.noise_variance * torch.eye(X.shape[0])
        
        # Compute inverse (with numerical stability)
        try:
            self.K_inv = torch.inverse(K)
        except:
            # Use pseudo-inverse if singular
            self.K_inv = torch.pinverse(K)
        
        self.alpha = torch.matmul(self.K_inv, y)
    
    def predict(self, X_test: torch.Tensor, return_std: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty."""
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Compute kernel matrices
        K_star = self.kernel_function(self.X_train, X_test)  # N x M
        K_star_star = self.kernel_function(X_test, X_test)   # M x M
        
        # Compute mean prediction
        mean = torch.matmul(K_star.T, self.alpha)
        
        if return_std:
            # Compute variance prediction
            v = torch.matmul(self.K_inv, K_star)
            variance = torch.diag(K_star_star) - torch.sum(K_star * v, dim=0)
            variance = torch.clamp(variance, min=1e-8)  # Ensure positive
            std = torch.sqrt(variance)
            return mean, std
        else:
            return mean, None
    
    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood for hyperparameter optimization."""
        if self.K_inv is None:
            return -np.inf
        
        K = self.kernel_function(self.X_train, self.X_train)
        K += self.noise_variance * torch.eye(self.X_train.shape[0])
        
        # Log marginal likelihood = -0.5 * (y^T K^{-1} y + log|K| + n*log(2π))
        sign, logdet = torch.slogdet(K)
        if sign <= 0:
            return -np.inf
        
        quad_form = torch.matmul(self.y_train.T, torch.matmul(self.K_inv, self.y_train))
        n = self.X_train.shape[0]
        
        return -0.5 * (quad_form + logdet + n * np.log(2 * np.pi))

class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network with variational inference."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 prior_std: float = 1.0):
        """Initialize Bayesian Neural Network."""
        super(BayesianNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prior_std = prior_std
        
        # Weight means
        self.fc1_mean = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, output_dim)
        
        # Weight log variances
        self.fc1_logvar = nn.Linear(input_dim, hidden_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, output_dim)
        
        # Initialize log variances to small values
        nn.init.constant_(self.fc1_logvar.weight, -3)
        nn.init.constant_(self.fc1_logvar.bias, -3)
        nn.init.constant_(self.fc2_logvar.weight, -3)
        nn.init.constant_(self.fc2_logvar.bias, -3)
        nn.init.constant_(self.fc3_logvar.weight, -3)
        nn.init.constant_(self.fc3_logvar.bias, -3)
        
    def forward(self, x, num_samples: int = 1):
        """Forward pass with weight sampling."""
        batch_size = x.shape[0]
        outputs = []
        
        for _ in range(num_samples):
            # Sample weights from variational distribution
            fc1_w_mean = self.fc1_mean.weight
            fc1_w_std = torch.exp(0.5 * self.fc1_logvar.weight)
            fc1_w = fc1_w_mean + fc1_w_std * torch.randn_like(fc1_w_mean)
            
            fc1_b_mean = self.fc1_mean.bias
            fc1_b_std = torch.exp(0.5 * self.fc1_logvar.bias)
            fc1_b = fc1_b_mean + fc1_b_std * torch.randn_like(fc1_b_mean)
            
            fc2_w_mean = self.fc2_mean.weight
            fc2_w_std = torch.exp(0.5 * self.fc2_logvar.weight)
            fc2_w = fc2_w_mean + fc2_w_std * torch.randn_like(fc2_w_mean)
            
            fc2_b_mean = self.fc2_mean.bias
            fc2_b_std = torch.exp(0.5 * self.fc2_logvar.bias)
            fc2_b = fc2_b_mean + fc2_b_std * torch.randn_like(fc2_b_mean)
            
            fc3_w_mean = self.fc3_mean.weight
            fc3_w_std = torch.exp(0.5 * self.fc3_logvar.weight)
            fc3_w = fc3_w_mean + fc3_w_std * torch.randn_like(fc3_w_mean)
            
            fc3_b_mean = self.fc3_mean.bias
            fc3_b_std = torch.exp(0.5 * self.fc3_logvar.bias)
            fc3_b = fc3_b_mean + fc3_b_std * torch.randn_like(fc3_b_mean)
            
            # Forward pass with sampled weights
            h1 = F.relu(F.linear(x, fc1_w, fc1_b))
            h2 = F.relu(F.linear(h1, fc2_w, fc2_b))
            output = F.linear(h2, fc3_w, fc3_b)
            
            outputs.append(output)
        
        return torch.stack(outputs)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between variational and prior distributions."""
        kl = 0.0
        
        # KL for each layer
        for mean_layer, logvar_layer in [(self.fc1_mean, self.fc1_logvar),
                                       (self.fc2_mean, self.fc2_logvar),
                                       (self.fc3_mean, self.fc3_logvar)]:
            # Weight KL
            mean_w = mean_layer.weight
            logvar_w = logvar_layer.weight
            kl += self._kl_gaussian(mean_w, logvar_w, 0.0, np.log(self.prior_std ** 2))
            
            # Bias KL
            mean_b = mean_layer.bias
            logvar_b = logvar_layer.bias
            kl += self._kl_gaussian(mean_b, logvar_b, 0.0, np.log(self.prior_std ** 2))
        
        return kl
    
    def _kl_gaussian(self, mean, logvar, prior_mean, prior_logvar):
        """KL divergence between two Gaussians."""
        return 0.5 * (prior_logvar - logvar + 
                     (torch.exp(logvar) + (mean - prior_mean) ** 2) / torch.exp(prior_logvar) - 1)

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for probabilistic representation learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """Initialize VAE."""
        super(VariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from prior distribution."""
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        return self.decode(z)

class GaussianMixtureModel:
    """Gaussian Mixture Model with EM algorithm."""
    
    def __init__(self, num_components: int, max_iter: int = 100, tol: float = 1e-6):
        """Initialize GMM."""
        self.num_components = num_components
        self.max_iter = max_iter
        self.tol = tol
        
        self.weights = None
        self.means = None
        self.covariances = None
        self.converged = False
        
    def fit(self, X: torch.Tensor):
        """Fit GMM using EM algorithm."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = torch.ones(self.num_components) / self.num_components
        
        # Initialize means with k-means++
        self.means = self._init_means(X)
        
        # Initialize covariances
        self.covariances = torch.stack([torch.eye(n_features) for _ in range(self.num_components)])
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            responsibilities = self._e_step(X)
            
            # M-step: update parameters
            self._m_step(X, responsibilities)
            
            # Check convergence
            log_likelihood = self._log_likelihood(X)
            
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged = True
                break
                
            prev_log_likelihood = log_likelihood
    
    def _init_means(self, X: torch.Tensor) -> torch.Tensor:
        """Initialize means using k-means++ algorithm."""
        n_samples, n_features = X.shape
        means = torch.zeros(self.num_components, n_features)
        
        # Choose first center randomly
        means[0] = X[torch.randint(0, n_samples, (1,))]
        
        for i in range(1, self.num_components):
            # Compute distances to nearest centers
            distances = torch.min(torch.cdist(X, means[:i]), dim=1)[0]
            
            # Choose next center with probability proportional to squared distance
            probs = distances ** 2
            probs /= probs.sum()
            
            idx = torch.multinomial(probs, 1)
            means[i] = X[idx]
        
        return means
    
    def _e_step(self, X: torch.Tensor) -> torch.Tensor:
        """E-step: compute responsibilities."""
        n_samples = X.shape[0]
        responsibilities = torch.zeros(n_samples, self.num_components)
        
        for k in range(self.num_components):
            # Compute log probability for component k
            diff = X - self.means[k]
            cov_inv = torch.inverse(self.covariances[k] + 1e-6 * torch.eye(X.shape[1]))
            
            quad_form = torch.sum(diff @ cov_inv * diff, dim=1)
            log_det = torch.logdet(self.covariances[k] + 1e-6 * torch.eye(X.shape[1]))
            
            log_prob = torch.log(self.weights[k] + 1e-8) - 0.5 * (log_det + quad_form + X.shape[1] * np.log(2 * np.pi))
            responsibilities[:, k] = log_prob
        
        # Convert to responsibilities (normalize)
        responsibilities = torch.softmax(responsibilities, dim=1)
        
        return responsibilities
    
    def _m_step(self, X: torch.Tensor, responsibilities: torch.Tensor):
        """M-step: update parameters."""
        n_samples = X.shape[0]
        
        # Update weights
        self.weights = responsibilities.sum(dim=0) / n_samples
        
        # Update means
        for k in range(self.num_components):
            self.means[k] = torch.sum(responsibilities[:, k:k+1] * X, dim=0) / responsibilities[:, k].sum()
        
        # Update covariances
        for k in range(self.num_components):
            diff = X - self.means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            self.covariances[k] = (weighted_diff.T @ diff) / responsibilities[:, k].sum()
            
            # Add regularization
            self.covariances[k] += 1e-6 * torch.eye(X.shape[1])
    
    def _log_likelihood(self, X: torch.Tensor) -> float:
        """Compute log likelihood."""
        log_likelihood = 0.0
        
        for k in range(self.num_components):
            diff = X - self.means[k]
            cov_inv = torch.inverse(self.covariances[k])
            
            quad_form = torch.sum(diff @ cov_inv * diff, dim=1)
            log_det = torch.logdet(self.covariances[k])
            
            log_prob = torch.log(self.weights[k] + 1e-8) - 0.5 * (log_det + quad_form + X.shape[1] * np.log(2 * np.pi))
            log_likelihood += torch.logsumexp(log_prob, dim=0)
        
        return log_likelihood.item()
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict component probabilities."""
        return self._e_step(X)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict component assignments."""
        responsibilities = self.predict_proba(X)
        return torch.argmax(responsibilities, dim=1)

class HiddenMarkovModel:
    """Hidden Markov Model for sequential data."""
    
    def __init__(self, num_states: int, num_observations: int):
        """Initialize HMM."""
        self.num_states = num_states
        self.num_observations = num_observations
        
        # Initialize parameters randomly
        self.initial_probs = torch.rand(num_states)
        self.initial_probs /= self.initial_probs.sum()
        
        self.transition_matrix = torch.rand(num_states, num_states)
        self.transition_matrix /= self.transition_matrix.sum(dim=1, keepdim=True)
        
        self.emission_matrix = torch.rand(num_states, num_observations)
        self.emission_matrix /= self.emission_matrix.sum(dim=1, keepdim=True)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Forward algorithm for computing likelihood."""
        T = len(observations)
        alpha = torch.zeros(T, self.num_states)
        
        # Initialize
        alpha[0] = self.initial_probs * self.emission_matrix[:, observations[0]]
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.num_states):
                alpha[t, j] = torch.sum(alpha[t-1] * self.transition_matrix[:, j]) * \
                             self.emission_matrix[j, observations[t]]
        
        # Total likelihood
        likelihood = torch.sum(alpha[-1])
        
        return alpha, likelihood.item()
    
    def backward(self, observations: torch.Tensor) -> torch.Tensor:
        """Backward algorithm."""
        T = len(observations)
        beta = torch.zeros(T, self.num_states)
        
        # Initialize
        beta[-1] = 1.0
        
        # Backward pass
        for t in range(T-2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = torch.sum(self.transition_matrix[i] * 
                                     self.emission_matrix[:, observations[t+1]] * 
                                     beta[t+1])
        
        return beta
    
    def viterbi(self, observations: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Viterbi algorithm for most likely state sequence."""
        T = len(observations)
        delta = torch.zeros(T, self.num_states)
        psi = torch.zeros(T, self.num_states, dtype=torch.long)
        
        # Initialize
        delta[0] = torch.log(self.initial_probs + 1e-8) + \
                  torch.log(self.emission_matrix[:, observations[0]] + 1e-8)
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.num_states):
                scores = delta[t-1] + torch.log(self.transition_matrix[:, j] + 1e-8)
                psi[t, j] = torch.argmax(scores)
                delta[t, j] = torch.max(scores) + torch.log(self.emission_matrix[j, observations[t]] + 1e-8)
        
        # Backtrack
        states = torch.zeros(T, dtype=torch.long)
        states[-1] = torch.argmax(delta[-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        # Path probability
        path_prob = torch.max(delta[-1]).item()
        
        return states, path_prob

class UncertaintyQuantification:
    """Uncertainty quantification utilities."""
    
    @staticmethod
    def epistemic_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
        """Compute epistemic (model) uncertainty."""
        # Variance across model samples
        return torch.var(predictions, dim=0)
    
    @staticmethod
    def aleatoric_uncertainty(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute aleatoric (data) uncertainty."""
        # Average residual variance
        residuals = predictions - targets.unsqueeze(0)
        return torch.mean(residuals ** 2, dim=0)
    
    @staticmethod
    def total_uncertainty(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute total uncertainty."""
        epistemic = UncertaintyQuantification.epistemic_uncertainty(predictions)
        aleatoric = UncertaintyQuantification.aleatoric_uncertainty(predictions, targets)
        return epistemic + aleatoric
    
    @staticmethod
    def confidence_intervals(predictions: torch.Tensor, confidence: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute confidence intervals."""
        alpha = 1 - confidence
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower = torch.quantile(predictions, lower_quantile, dim=0)
        upper = torch.quantile(predictions, upper_quantile, dim=0)
        
        return lower, upper
    
    @staticmethod
    def prediction_intervals(mean: torch.Tensor, std: torch.Tensor, 
                           confidence: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute prediction intervals assuming Gaussian distribution."""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return lower, upper

class ProbabilisticModelEngine:
    """Main engine for probabilistic machine learning."""
    
    def __init__(self):
        """Initialize probabilistic model engine."""
        self.models = {}
        self.training_history = {}
        self.device = DEVICE
        
    def create_gaussian_process(self, model_name: str, kernel_type: KernelType = KernelType.RBF,
                              **kernel_params) -> GaussianProcess:
        """Create and register Gaussian Process."""
        gp = GaussianProcess(kernel_type=kernel_type, **kernel_params)
        self.models[model_name] = gp
        return gp
    
    def create_bayesian_nn(self, model_name: str, config: ProbabilisticConfig) -> BayesianNeuralNetwork:
        """Create and register Bayesian Neural Network."""
        bnn = BayesianNeuralNetwork(
            config.input_dim, config.hidden_dim, config.output_dim, config.prior_std
        ).to(self.device)
        
        self.models[model_name] = bnn
        return bnn
    
    def create_vae(self, model_name: str, input_dim: int, hidden_dim: int, 
                   latent_dim: int) -> VariationalAutoencoder:
        """Create and register Variational Autoencoder."""
        vae = VariationalAutoencoder(input_dim, hidden_dim, latent_dim).to(self.device)
        self.models[model_name] = vae
        return vae
    
    def create_gmm(self, model_name: str, num_components: int, **kwargs) -> GaussianMixtureModel:
        """Create and register Gaussian Mixture Model."""
        gmm = GaussianMixtureModel(num_components, **kwargs)
        self.models[model_name] = gmm
        return gmm
    
    def create_hmm(self, model_name: str, num_states: int, num_observations: int) -> HiddenMarkovModel:
        """Create and register Hidden Markov Model."""
        hmm = HiddenMarkovModel(num_states, num_observations)
        self.models[model_name] = hmm
        return hmm
    
    def train_bayesian_nn(self, model_name: str, X_train: torch.Tensor, y_train: torch.Tensor,
                         config: ProbabilisticConfig, X_val: Optional[torch.Tensor] = None,
                         y_val: Optional[torch.Tensor] = None) -> Dict:
        """Train Bayesian Neural Network."""
        
        if model_name not in self.models or not isinstance(self.models[model_name], BayesianNeuralNetwork):
            raise ValueError(f"Bayesian NN {model_name} not found")
        
        model = self.models[model_name]
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
        
        training_history = []
        
        for epoch in range(config.num_epochs):
            model.train()
            
            # Mini-batch training
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train), config.batch_size):
                batch_X = X_train[i:i+config.batch_size]
                batch_y = y_train[i:i+config.batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass with multiple samples
                predictions = model(batch_X, num_samples=config.mc_samples)  # (samples, batch, output)
                mean_pred = predictions.mean(dim=0)
                
                # Likelihood loss (reconstruction)
                likelihood_loss = F.mse_loss(mean_pred, batch_y)
                
                # KL divergence loss
                kl_loss = model.kl_divergence().sum()
                
                # Total loss
                loss = likelihood_loss + config.kl_weight * kl_loss / len(batch_X)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validation
            val_loss = 0.0
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_predictions = model(X_val, num_samples=config.mc_samples)
                    val_mean = val_predictions.mean(dim=0)
                    val_loss = F.mse_loss(val_mean, y_val).item()
            
            epoch_info = {
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'kl_weight': config.kl_weight
            }
            
            training_history.append(epoch_info)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={avg_loss:.6f}, Val Loss={val_loss:.6f}")
        
        self.training_history[model_name] = training_history
        return training_history
    
    def train_vae(self, model_name: str, X_train: torch.Tensor, config: ProbabilisticConfig,
                  X_val: Optional[torch.Tensor] = None) -> Dict:
        """Train Variational Autoencoder."""
        
        if model_name not in self.models or not isinstance(self.models[model_name], VariationalAutoencoder):
            raise ValueError(f"VAE {model_name} not found")
        
        model = self.models[model_name]
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        X_train = X_train.to(self.device)
        if X_val is not None:
            X_val = X_val.to(self.device)
        
        training_history = []
        
        for epoch in range(config.num_epochs):
            model.train()
            
            total_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train), config.batch_size):
                batch_X = X_train[i:i+config.batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                recon_x, mu, logvar = model(batch_X)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(recon_x, batch_X, reduction='sum')
                
                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Total loss
                loss = recon_loss + config.kl_weight * kl_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / len(X_train)
            avg_recon_loss = total_recon_loss / len(X_train)
            avg_kl_loss = total_kl_loss / len(X_train)
            
            # Validation
            val_loss = 0.0
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_recon, val_mu, val_logvar = model(X_val)
                    val_recon_loss = F.mse_loss(val_recon, X_val, reduction='sum')
                    val_kl_loss = -0.5 * torch.sum(1 + val_logvar - val_mu.pow(2) - val_logvar.exp())
                    val_loss = (val_recon_loss + config.kl_weight * val_kl_loss).item() / len(X_val)
            
            epoch_info = {
                'epoch': epoch,
                'total_loss': avg_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss,
                'val_loss': val_loss
            }
            
            training_history.append(epoch_info)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Total={avg_loss:.4f}, Recon={avg_recon_loss:.4f}, "
                      f"KL={avg_kl_loss:.4f}, Val={val_loss:.4f}")
        
        self.training_history[model_name] = training_history
        return training_history
    
    def predict_with_uncertainty(self, model_name: str, X_test: torch.Tensor,
                               num_samples: int = 100) -> Dict:
        """Make predictions with uncertainty quantification."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        X_test = X_test.to(self.device)
        
        if isinstance(model, GaussianProcess):
            mean, std = model.predict(X_test, return_std=True)
            
            # Generate samples from predictive distribution
            if std is not None:
                samples = []
                for i in range(num_samples):
                    sample = torch.normal(mean, std)
                    samples.append(sample)
                predictions = torch.stack(samples)
            else:
                predictions = mean.unsqueeze(0).repeat(num_samples, 1)
            
        elif isinstance(model, BayesianNeuralNetwork):
            model.eval()
            with torch.no_grad():
                predictions = model(X_test, num_samples=num_samples)
            
        else:
            raise ValueError(f"Uncertainty prediction not supported for {type(model)}")
        
        # Compute uncertainty metrics
        mean_pred = predictions.mean(dim=0)
        epistemic_unc = UncertaintyQuantification.epistemic_uncertainty(predictions)
        
        # Confidence intervals
        lower_ci, upper_ci = UncertaintyQuantification.confidence_intervals(predictions)
        
        return {
            'predictions': predictions,
            'mean': mean_pred,
            'epistemic_uncertainty': epistemic_unc,
            'confidence_interval_lower': lower_ci,
            'confidence_interval_upper': upper_ci
        }
    
    def get_model_summary(self, model_name: str) -> Dict:
        """Get model summary and statistics."""
        
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        summary = {
            'model_type': type(model).__name__,
            'model_name': model_name
        }
        
        if isinstance(model, GaussianProcess):
            summary.update({
                'kernel_type': model.kernel_type.value,
                'length_scale': model.length_scale,
                'variance': model.variance,
                'noise_variance': model.noise_variance,
                'training_size': model.X_train.shape[0] if model.X_train is not None else 0
            })
        
        elif isinstance(model, BayesianNeuralNetwork):
            total_params = sum(p.numel() for p in model.parameters())
            summary.update({
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'output_dim': model.output_dim,
                'total_parameters': total_params,
                'prior_std': model.prior_std
            })
        
        elif isinstance(model, VariationalAutoencoder):
            total_params = sum(p.numel() for p in model.parameters())
            summary.update({
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'latent_dim': model.latent_dim,
                'total_parameters': total_params
            })
        
        elif isinstance(model, GaussianMixtureModel):
            summary.update({
                'num_components': model.num_components,
                'converged': model.converged,
                'weights': model.weights.tolist() if model.weights is not None else None
            })
        
        elif isinstance(model, HiddenMarkovModel):
            summary.update({
                'num_states': model.num_states,
                'num_observations': model.num_observations
            })
        
        # Add training history if available
        if model_name in self.training_history:
            history = self.training_history[model_name]
            if history:
                summary['training_epochs'] = len(history)
                summary['final_train_loss'] = history[-1].get('train_loss', history[-1].get('total_loss'))
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    print("Testing Probabilistic Machine Learning Models...")
    
    # Generate synthetic data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate 1D regression data
    n_samples = 200
    X_1d = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y_1d = torch.sin(X_1d.squeeze()) + 0.1 * torch.randn(n_samples)
    
    # Generate high-dimensional data
    n_features = 10
    X_hd = torch.randn(n_samples, n_features)
    y_hd = torch.sum(X_hd[:, :3], dim=1) + 0.1 * torch.randn(n_samples)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train_1d, X_test_1d = X_1d[:train_size], X_1d[train_size:]
    y_train_1d, y_test_1d = y_1d[:train_size], y_1d[train_size:]
    
    X_train_hd, X_test_hd = X_hd[:train_size], X_hd[train_size:]
    y_train_hd, y_test_hd = y_hd[:train_size], y_hd[train_size:]
    
    # Initialize engine
    engine = ProbabilisticModelEngine()
    
    # Test Gaussian Process
    print("\n=== Testing Gaussian Process ===")
    
    gp_rbf = engine.create_gaussian_process("gp_rbf", KernelType.RBF, length_scale=0.5, variance=1.0)
    gp_rbf.fit(X_train_1d, y_train_1d)
    
    # Make predictions with uncertainty
    mean_pred, std_pred = gp_rbf.predict(X_test_1d, return_std=True)
    
    print(f"GP RBF - Mean prediction error: {F.mse_loss(mean_pred, y_test_1d):.6f}")
    print(f"GP RBF - Average uncertainty: {std_pred.mean():.6f}")
    print(f"GP RBF - Log marginal likelihood: {gp_rbf.log_marginal_likelihood():.4f}")
    
    # Test different kernels
    for kernel_type in [KernelType.MATERN, KernelType.PERIODIC]:
        gp = engine.create_gaussian_process(f"gp_{kernel_type.value}", kernel_type)
        gp.fit(X_train_1d, y_train_1d)
        mean_pred, _ = gp.predict(X_test_1d)
        mse = F.mse_loss(mean_pred, y_test_1d)
        print(f"GP {kernel_type.value} - MSE: {mse:.6f}")
    
    # Test Bayesian Neural Network
    print("\n=== Testing Bayesian Neural Network ===")
    
    bnn_config = ProbabilisticConfig(
        model_type=ProbabilisticModelType.BAYESIAN_NEURAL_NETWORK,
        input_dim=n_features,
        output_dim=1,
        hidden_dim=64,
        num_components=1,
        learning_rate=0.01,
        num_epochs=50,
        batch_size=32,
        prior_std=1.0,
        likelihood_std=0.1,
        kl_weight=0.1,
        mc_samples=10
    )
    
    bnn = engine.create_bayesian_nn("bnn_regressor", bnn_config)
    
    # Train BNN
    train_history = engine.train_bayesian_nn(
        "bnn_regressor", X_train_hd, y_train_hd.unsqueeze(1), bnn_config,
        X_test_hd, y_test_hd.unsqueeze(1)
    )
    
    # Make predictions with uncertainty
    bnn_results = engine.predict_with_uncertainty("bnn_regressor", X_test_hd, num_samples=50)
    bnn_mean = bnn_results['mean'].squeeze()
    bnn_uncertainty = bnn_results['epistemic_uncertainty'].squeeze()
    
    print(f"BNN - Mean prediction error: {F.mse_loss(bnn_mean, y_test_hd):.6f}")
    print(f"BNN - Average uncertainty: {bnn_uncertainty.mean():.6f}")
    print(f"BNN - Final training loss: {train_history[-1]['train_loss']:.6f}")
    
    # Test Variational Autoencoder
    print("\n=== Testing Variational Autoencoder ===")
    
    vae_config = ProbabilisticConfig(
        model_type=ProbabilisticModelType.VARIATIONAL_AUTOENCODER,
        input_dim=n_features,
        output_dim=n_features,
        hidden_dim=32,
        num_components=1,
        learning_rate=0.001,
        num_epochs=50,
        batch_size=32,
        prior_std=1.0,
        likelihood_std=1.0,
        kl_weight=1.0,
        mc_samples=1
    )
    
    vae = engine.create_vae("vae_model", n_features, 32, 8)
    
    # Train VAE
    vae_history = engine.train_vae("vae_model", X_train_hd, vae_config, X_test_hd)
    
    # Test reconstruction and generation
    vae.eval()
    with torch.no_grad():
        recon_x, mu, logvar = vae(X_test_hd.to(DEVICE))
        recon_error = F.mse_loss(recon_x, X_test_hd.to(DEVICE))
        
        # Generate new samples
        generated_samples = vae.sample(10)
    
    print(f"VAE - Reconstruction error: {recon_error:.6f}")
    print(f"VAE - Final total loss: {vae_history[-1]['total_loss']:.6f}")
    print(f"VAE - Generated sample shape: {generated_samples.shape}")
    
    # Test Gaussian Mixture Model
    print("\n=== Testing Gaussian Mixture Model ===")
    
    # Generate mixture data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create 3 clusters
    cluster1 = torch.randn(50, 2) + torch.tensor([2, 2])
    cluster2 = torch.randn(50, 2) + torch.tensor([-2, 2])
    cluster3 = torch.randn(50, 2) + torch.tensor([0, -2])
    mixture_data = torch.cat([cluster1, cluster2, cluster3], dim=0)
    
    gmm = engine.create_gmm("gmm_3comp", num_components=3, max_iter=50)
    gmm.fit(mixture_data)
    
    # Predict cluster assignments
    cluster_probs = gmm.predict_proba(mixture_data)
    cluster_assignments = gmm.predict(mixture_data)
    
    print(f"GMM - Converged: {gmm.converged}")
    print(f"GMM - Component weights: {gmm.weights.numpy()}")
    print(f"GMM - Average cluster probability: {cluster_probs.max(dim=1)[0].mean():.6f}")
    
    # Test Hidden Markov Model
    print("\n=== Testing Hidden Markov Model ===")
    
    hmm = engine.create_hmm("hmm_weather", num_states=2, num_observations=3)
    
    # Generate synthetic observation sequence
    observations = torch.randint(0, 3, (20,))
    
    # Forward algorithm
    alpha, likelihood = hmm.forward(observations)
    print(f"HMM - Sequence likelihood: {likelihood:.6f}")
    
    # Viterbi algorithm
    states, path_prob = hmm.viterbi(observations)
    print(f"HMM - Most likely state sequence: {states.tolist()}")
    print(f"HMM - Path probability: {path_prob:.6f}")
    
    # Test Uncertainty Quantification
    print("\n=== Testing Uncertainty Quantification ===")
    
    # Generate multiple predictions for uncertainty analysis
    num_models = 20
    predictions = []
    
    for i in range(num_models):
        noise = 0.1 * torch.randn_like(y_test_hd)
        pred = y_test_hd + noise
        predictions.append(pred)
    
    predictions = torch.stack(predictions)
    
    # Compute uncertainties
    epistemic_unc = UncertaintyQuantification.epistemic_uncertainty(predictions)
    aleatoric_unc = UncertaintyQuantification.aleatoric_uncertainty(predictions, y_test_hd)
    total_unc = UncertaintyQuantification.total_uncertainty(predictions, y_test_hd)
    
    # Confidence intervals
    lower_ci, upper_ci = UncertaintyQuantification.confidence_intervals(predictions, confidence=0.95)
    
    print(f"Average epistemic uncertainty: {epistemic_unc.mean():.6f}")
    print(f"Average aleatoric uncertainty: {aleatoric_unc.mean():.6f}")
    print(f"Average total uncertainty: {total_unc.mean():.6f}")
    print(f"CI coverage: {((y_test_hd >= lower_ci) & (y_test_hd <= upper_ci)).float().mean():.3f}")
    
    # Model summaries
    print("\n=== Model Summaries ===")
    for model_name in engine.models.keys():
        summary = engine.get_model_summary(model_name)
        print(f"\n{model_name}:")
        for key, value in summary.items():
            if key != 'model_name':
                print(f"  {key}: {value}")
    
    print("\nProbabilistic machine learning models testing completed successfully!")