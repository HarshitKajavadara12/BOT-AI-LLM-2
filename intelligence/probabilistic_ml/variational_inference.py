"""
Variational Inference for Financial Models
Advanced VI implementations for Bayesian deep learning and probabilistic modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, Categorical
from torch.distributions.kl import kl_divergence
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from abc import ABC, abstractmethod
import math


class VariationalDistribution(ABC):
    """Base class for variational distributions"""
    
    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from the variational distribution"""
        pass
    
    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of value"""
        pass
    
    @abstractmethod
    def kl_divergence(self, prior) -> torch.Tensor:
        """Compute KL divergence with prior"""
        pass
    
    @abstractmethod
    def mean(self) -> torch.Tensor:
        """Mean of the distribution"""
        pass


class MeanFieldGaussian(VariationalDistribution):
    """
    Mean-field Gaussian variational distribution
    """
    
    def __init__(self, shape: Tuple[int, ...], init_std: float = 0.1):
        self.shape = shape
        
        # Variational parameters
        self.mu = nn.Parameter(torch.randn(shape) * 0.01)
        self.log_sigma = nn.Parameter(torch.log(torch.full(shape, init_std)))
        
    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample using reparameterization trick"""
        eps = torch.randn(sample_shape + self.shape, device=self.mu.device)
        return self.mu + self.sigma * eps
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability"""
        var = self.sigma ** 2
        log_prob = -0.5 * ((value - self.mu) ** 2 / var + torch.log(2 * math.pi * var))
        return log_prob.sum(dim=-1)  # Sum over parameter dimensions
    
    def kl_divergence(self, prior_mean: float = 0.0, prior_std: float = 1.0) -> torch.Tensor:
        """KL divergence with Gaussian prior"""
        prior_var = prior_std ** 2
        var = self.sigma ** 2
        
        kl = 0.5 * (
            torch.log(prior_var / var) +
            var / prior_var +
            ((self.mu - prior_mean) ** 2) / prior_var - 1
        )
        return kl.sum()
    
    def mean(self) -> torch.Tensor:
        return self.mu


class FullCovarianceGaussian(VariationalDistribution):
    """
    Full covariance Gaussian variational distribution
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        
        # Mean parameter
        self.mu = nn.Parameter(torch.randn(dim) * 0.01)
        
        # Cholesky decomposition of covariance matrix
        self.L_tril = nn.Parameter(torch.eye(dim) * 0.1)
        
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample using multivariate normal"""
        dist = MultivariateNormal(self.mu, scale_tril=self.L_tril)
        return dist.sample(sample_shape)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability"""
        dist = MultivariateNormal(self.mu, scale_tril=self.L_tril)
        return dist.log_prob(value)
    
    def kl_divergence(self, prior_mean: torch.Tensor, prior_cov: torch.Tensor) -> torch.Tensor:
        """KL divergence with multivariate Gaussian prior"""
        q_dist = MultivariateNormal(self.mu, scale_tril=self.L_tril)
        p_dist = MultivariateNormal(prior_mean, prior_cov)
        return kl_divergence(q_dist, p_dist)
    
    def mean(self) -> torch.Tensor:
        return self.mu


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with variational inference
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        init_std: float = 0.1
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.weight_log_sigma = nn.Parameter(torch.log(torch.full((out_features, in_features), init_std)))
        
        # Variational parameters for bias
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.01)
        self.bias_log_sigma = nn.Parameter(torch.log(torch.full((out_features,), init_std)))
        
    @property
    def weight_sigma(self) -> torch.Tensor:
        return torch.exp(self.weight_log_sigma)
    
    @property
    def bias_sigma(self) -> torch.Tensor:
        return torch.exp(self.bias_log_sigma)
    
    def sample_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample weights and bias"""
        # Sample weights
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + self.weight_sigma * weight_eps
        
        # Sample bias
        bias_eps = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + self.bias_sigma * bias_eps
        
        return weight, bias
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass"""
        if sample:
            weight, bias = self.sample_weights()
        else:
            # Use mean parameters
            weight, bias = self.weight_mu, self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """KL divergence with prior"""
        # KL for weights
        weight_var = self.weight_sigma ** 2
        weight_kl = 0.5 * (
            torch.log(self.prior_std ** 2 / weight_var) +
            weight_var / (self.prior_std ** 2) +
            (self.weight_mu ** 2) / (self.prior_std ** 2) - 1
        ).sum()
        
        # KL for bias
        bias_var = self.bias_sigma ** 2
        bias_kl = 0.5 * (
            torch.log(self.prior_std ** 2 / bias_var) +
            bias_var / (self.prior_std ** 2) +
            (self.bias_mu ** 2) / (self.prior_std ** 2) - 1
        ).sum()
        
        return weight_kl + bias_kl


class BayesianMLP(nn.Module):
    """
    Bayesian Multi-Layer Perceptron
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        prior_std: float = 1.0,
        init_std: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.activation = getattr(F, activation)
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            layer = BayesianLinear(dims[i], dims[i+1], prior_std, init_std)
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass"""
        for i, layer in enumerate(self.layers):
            x = layer(x, sample=sample)
            
            # Apply activation (except for last layer)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence"""
        total_kl = 0
        for layer in self.layers:
            total_kl += layer.kl_divergence()
        return total_kl
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty quantification
        
        Args:
            x: Input data [batch_size, input_dim]
            n_samples: Number of Monte Carlo samples
        
        Returns:
            mean: Predictive mean [batch_size, output_dim]
            std: Predictive standard deviation [batch_size, output_dim]
        """
        self.eval()
        
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x, sample=True)
                samples.append(pred)
        
        samples = torch.stack(samples, dim=0)  # [n_samples, batch_size, output_dim]
        
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        return mean, std


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for financial data representation learning
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [256, 128],
        beta: float = 1.0
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        encoder_dims = [input_dim] + hidden_dims
        encoder_layers = []
        
        for i in range(len(encoder_dims) - 1):
            encoder_layers.extend([
                nn.Linear(encoder_dims[i], encoder_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        decoder_layers = []
        
        for i in range(len(decoder_dims) - 2):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        # Final layer without activation
        decoder_layers.append(nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent variable to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
    
    def loss_function(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss"""
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def generate(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        """Generate new samples"""
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)


class VariationalRNN(nn.Module):
    """
    Variational Recurrent Neural Network for time series
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,
        num_layers: int = 1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder RNN
        self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def encode_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence to latent parameters"""
        # Run encoder RNN
        rnn_out, (hidden, cell) = self.encoder_rnn(x)
        
        # Use final hidden state
        final_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Get latent parameters
        mu = self.fc_mu(final_hidden)
        log_var = self.fc_log_var(final_hidden)
        
        return mu, log_var
    
    def decode_sequence(
        self,
        z: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """Decode latent variable to sequence"""
        batch_size = z.size(0)
        
        # Repeat latent variable for each time step
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Run decoder RNN
        rnn_out, _ = self.decoder_rnn(z_seq)
        
        # Generate output sequence
        output = self.output_layer(rnn_out)
        
        return output
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        seq_len = x.size(1)
        
        # Encode
        mu, log_var = self.encode_sequence(x)
        
        # Reparameterize
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Decode
        reconstruction = self.decode_sequence(z, seq_len)
        
        return reconstruction, mu, log_var


class ConditionalVariationalAutoencoder(nn.Module):
    """
    Conditional VAE for market regime-aware modeling
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder (input + condition)
        encoder_input_dim = input_dim + condition_dim
        encoder_dims = [encoder_input_dim] + hidden_dims
        
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.extend([
                nn.Linear(encoder_dims[i], encoder_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (latent + condition)
        decoder_input_dim = latent_dim + condition_dim
        decoder_dims = [decoder_input_dim] + hidden_dims[::-1] + [input_dim]
        
        decoder_layers = []
        for i in range(len(decoder_dims) - 2):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        decoder_layers.append(nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with condition"""
        # Concatenate input and condition
        encoder_input = torch.cat([x, condition], dim=-1)
        
        h = self.encoder(encoder_input)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        return mu, log_var
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode with condition"""
        # Concatenate latent and condition
        decoder_input = torch.cat([z, condition], dim=-1)
        return self.decoder(decoder_input)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, log_var = self.encode(x, condition)
        
        # Reparameterize
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        reconstruction = self.decode(z, condition)
        
        return reconstruction, mu, log_var


class StochasticVariationalInference:
    """
    Stochastic Variational Inference optimizer
    """
    
    def __init__(
        self,
        model: nn.Module,
        likelihood_fn: Callable,
        optimizer: torch.optim.Optimizer,
        n_samples: int = 10,
        kl_weight: float = 1.0
    ):
        self.model = model
        self.likelihood_fn = likelihood_fn
        self.optimizer = optimizer
        self.n_samples = n_samples
        self.kl_weight = kl_weight
    
    def compute_elbo(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Evidence Lower BOund (ELBO)
        
        Args:
            x: Input data
            y: Target data
        
        Returns:
            elbo: Evidence lower bound
            metrics: Dictionary of loss components
        """
        
        # Monte Carlo samples
        log_likelihoods = []
        
        for _ in range(self.n_samples):
            # Forward pass with sampling
            pred = self.model(x, sample=True)
            
            # Compute log likelihood
            log_likelihood = self.likelihood_fn(pred, y)
            log_likelihoods.append(log_likelihood)
        
        # Average log likelihood
        avg_log_likelihood = torch.stack(log_likelihoods).mean(dim=0)
        
        # KL divergence
        if hasattr(self.model, 'kl_divergence'):
            kl_div = self.model.kl_divergence()
        else:
            kl_div = torch.tensor(0.0, device=x.device)
        
        # ELBO = E[log p(y|x, θ)] - KL[q(θ) || p(θ)]
        elbo = avg_log_likelihood.sum() - self.kl_weight * kl_div
        
        metrics = {
            'elbo': elbo,
            'log_likelihood': avg_log_likelihood.sum(),
            'kl_divergence': kl_div,
            'loss': -elbo  # Negative ELBO for minimization
        }
        
        return -elbo, metrics  # Return negative for minimization
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, metrics = self.compute_elbo(x, y)
        loss.backward()
        self.optimizer.step()
        
        # Convert to float for logging
        float_metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
        
        return float_metrics


def train_variational_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    kl_weight: float = 1.0,
    device: torch.device = None
) -> Dict[str, List[float]]:
    """
    Train variational model
    
    Args:
        model: Variational model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        lr: Learning rate
        kl_weight: Weight for KL divergence term
        device: Training device
    
    Returns:
        Training history
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Likelihood function (assuming regression)
    def likelihood_fn(pred, target):
        return -0.5 * ((pred - target) ** 2).sum(dim=-1)  # Negative MSE
    
    # VI optimizer
    vi_optimizer = StochasticVariationalInference(
        model, likelihood_fn, optimizer, kl_weight=kl_weight
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_elbo': [],
        'train_kl': [],
        'val_loss': [],
        'val_elbo': [],
        'val_kl': []
    }
    
    for epoch in range(n_epochs):
        # Training
        train_metrics = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            metrics = vi_optimizer.train_step(batch_x, batch_y)
            train_metrics.append(metrics)
        
        # Average training metrics
        avg_train_metrics = {
            k: np.mean([m[k] for m in train_metrics])
            for k in train_metrics[0].keys()
        }
        
        history['train_loss'].append(avg_train_metrics['loss'])
        history['train_elbo'].append(avg_train_metrics['elbo'])
        history['train_kl'].append(avg_train_metrics['kl_divergence'])
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_metrics = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    loss, metrics = vi_optimizer.compute_elbo(batch_x, batch_y)
                    float_metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
                    val_metrics.append(float_metrics)
            
            avg_val_metrics = {
                k: np.mean([m[k] for m in val_metrics])
                for k in val_metrics[0].keys()
            }
            
            history['val_loss'].append(avg_val_metrics['loss'])
            history['val_elbo'].append(avg_val_metrics['elbo'])
            history['val_kl'].append(avg_val_metrics['kl_divergence'])
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {avg_train_metrics['loss']:.4f}")
            if val_loader is not None:
                print(f"  Val Loss: {avg_val_metrics['loss']:.4f}")
    
    return history


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    
    print("Testing Bayesian Neural Network...")
    
    # Generate synthetic data
    n_samples = 1000
    input_dim = 10
    output_dim = 1
    
    X = torch.randn(n_samples, input_dim)
    y = torch.sum(X[:, :3], dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1
    
    # Create Bayesian MLP
    model = BayesianMLP(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        output_dim=output_dim,
        prior_std=1.0
    )
    
    # Test forward pass
    pred = model(X[:10])
    print(f"Prediction shape: {pred.shape}")
    
    # Test uncertainty quantification
    mean_pred, std_pred = model.predict_with_uncertainty(X[:10], n_samples=50)
    print(f"Predictive mean shape: {mean_pred.shape}")
    print(f"Predictive std shape: {std_pred.shape}")
    print(f"Average uncertainty: {std_pred.mean().item():.4f}")
    
    print("\nTesting Variational Autoencoder...")
    
    # Test VAE
    vae = VariationalAutoencoder(
        input_dim=input_dim,
        latent_dim=5,
        hidden_dims=[32, 16]
    )
    
    # Forward pass
    reconstruction, mu, log_var = vae(X[:10])
    
    # Compute loss
    loss_dict = vae.loss_function(reconstruction, X[:10], mu, log_var)
    print(f"VAE Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Reconstruction Loss: {loss_dict['reconstruction_loss'].item():.4f}")
    print(f"KL Loss: {loss_dict['kl_loss'].item():.4f}")
    
    # Generate samples
    generated = vae.generate(5)
    print(f"Generated samples shape: {generated.shape}")
    
    print("\nDone!")