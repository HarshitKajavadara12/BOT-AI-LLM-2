"""
Feature Learning and Representation Systems for QUANTUM-FORGE
Implements advanced feature learning techniques including autoencoders, manifold learning, 
representation learning, and transfer learning for quantitative trading.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import warnings
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import pickle
warnings.filterwarnings('ignore')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureLearningType(Enum):
    """Types of feature learning methods."""
    AUTOENCODER = "autoencoder"
    VARIATIONAL_AUTOENCODER = "variational_autoencoder"
    DENOISING_AUTOENCODER = "denoising_autoencoder"
    SPARSE_AUTOENCODER = "sparse_autoencoder"
    CONTRACTIVE_AUTOENCODER = "contractive_autoencoder"
    BETA_VAE = "beta_vae"
    WAE = "wasserstein_autoencoder"

class ManifoldLearningType(Enum):
    """Types of manifold learning methods."""
    PCA = "principal_component_analysis"
    ICA = "independent_component_analysis"
    TSNE = "t_distributed_stochastic_neighbor_embedding"
    ISOMAP = "isomap"
    LLE = "locally_linear_embedding"
    NMF = "non_negative_matrix_factorization"
    FACTOR_ANALYSIS = "factor_analysis"
    UMAP = "uniform_manifold_approximation"

class TransferLearningType(Enum):
    """Types of transfer learning approaches."""
    FEATURE_EXTRACTION = "feature_extraction"
    FINE_TUNING = "fine_tuning"
    DOMAIN_ADAPTATION = "domain_adaptation"
    MULTITASK_LEARNING = "multitask_learning"
    META_LEARNING = "meta_learning"

@dataclass
class FeatureLearningConfig:
    """Configuration for feature learning models."""
    method_type: FeatureLearningType
    input_dim: int
    latent_dim: int
    hidden_dims: List[int]
    learning_rate: float
    num_epochs: int
    batch_size: int
    dropout_rate: float
    regularization_weight: float
    noise_factor: float  # For denoising
    sparsity_weight: float  # For sparse autoencoder
    beta: float  # For β-VAE
    contraction_weight: float  # For contractive autoencoder

class BaseAutoencoder(nn.Module):
    """Base autoencoder class."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int], 
                 dropout_rate: float = 0.1):
        """Initialize base autoencoder."""
        super(BaseAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation for last layer
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(dropout_rate))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        dims_reversed = dims[::-1]
        
        for i in range(len(dims_reversed) - 1):
            decoder_layers.append(nn.Linear(dims_reversed[i], dims_reversed[i+1]))
            if i < len(dims_reversed) - 2:  # No activation for last layer
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to output."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z

class DenoisingAutoencoder(BaseAutoencoder):
    """Denoising Autoencoder."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int],
                 noise_factor: float = 0.3, dropout_rate: float = 0.1):
        """Initialize denoising autoencoder."""
        super().__init__(input_dim, latent_dim, hidden_dims, dropout_rate)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        """Add noise to input."""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise
    
    def forward(self, x, add_noise: bool = True):
        """Forward pass with optional noise."""
        if add_noise and self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
        
        z = self.encode(x_noisy)
        recon_x = self.decode(z)
        return recon_x, z

class SparseAutoencoder(BaseAutoencoder):
    """Sparse Autoencoder with L1 regularization on latent codes."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int],
                 sparsity_weight: float = 0.01, dropout_rate: float = 0.1):
        """Initialize sparse autoencoder."""
        super().__init__(input_dim, latent_dim, hidden_dims, dropout_rate)
        self.sparsity_weight = sparsity_weight
    
    def forward(self, x):
        """Forward pass through sparse autoencoder."""
        z = self.encode(x)
        recon_x = self.decode(z)
        
        # Compute sparsity penalty
        sparsity_loss = self.sparsity_weight * torch.mean(torch.abs(z))
        
        return recon_x, z, sparsity_loss

class ContractiveAutoencoder(BaseAutoencoder):
    """Contractive Autoencoder with Jacobian regularization."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int],
                 contraction_weight: float = 0.01, dropout_rate: float = 0.1):
        """Initialize contractive autoencoder."""
        super().__init__(input_dim, latent_dim, hidden_dims, dropout_rate)
        self.contraction_weight = contraction_weight
    
    def compute_jacobian_penalty(self, x, z):
        """Compute Jacobian penalty for contractive regularization."""
        # Enable gradient computation for input
        x_var = x.clone().detach().requires_grad_(True)
        
        # Forward pass to get latent representation
        z_var = self.encode(x_var)
        
        # Compute Jacobian penalty
        jacobian_penalty = 0.0
        
        for i in range(z.shape[1]):  # For each latent dimension
            # Compute gradients of i-th latent unit w.r.t. input
            if x_var.grad is not None:
                x_var.grad.zero_()
            
            z_var[:, i].sum().backward(retain_graph=True)
            
            if x_var.grad is not None:
                jacobian_penalty += torch.sum(x_var.grad ** 2)
        
        return self.contraction_weight * jacobian_penalty / x.shape[0]
    
    def forward(self, x):
        """Forward pass through contractive autoencoder."""
        z = self.encode(x)
        recon_x = self.decode(z)
        
        # Compute contractive penalty
        if self.training:
            contractive_loss = self.compute_jacobian_penalty(x, z)
        else:
            contractive_loss = torch.tensor(0.0).to(x.device)
        
        return recon_x, z, contractive_loss

class BetaVAE(nn.Module):
    """β-VAE for disentangled representation learning."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int],
                 beta: float = 4.0, dropout_rate: float = 0.1):
        """Initialize β-VAE."""
        super(BetaVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        encoder_layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        dims_reversed = [latent_dim] + hidden_dims[::-1] + [input_dim]
        
        for i in range(len(dims_reversed) - 1):
            decoder_layers.append(nn.Linear(dims_reversed[i], dims_reversed[i+1]))
            if i < len(dims_reversed) - 2:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
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
        """Forward pass through β-VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

class WassersteinAutoencoder(nn.Module):
    """Wasserstein Autoencoder (WAE) for stable training."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int],
                 regularization_weight: float = 10.0, dropout_rate: float = 0.1):
        """Initialize WAE."""
        super(WassersteinAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.regularization_weight = regularization_weight
        
        # Encoder
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(dropout_rate))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        dims_reversed = dims[::-1]
        
        for i in range(len(dims_reversed) - 1):
            decoder_layers.append(nn.Linear(dims_reversed[i], dims_reversed[i+1]))
            if i < len(dims_reversed) - 2:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through WAE."""
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z

class ManifoldLearner:
    """Manifold learning utilities."""
    
    def __init__(self):
        """Initialize manifold learner."""
        self.methods = {}
        self.fitted_transformers = {}
        self.embedding_results = {}
    
    def fit_pca(self, X: np.ndarray, n_components: int = None, 
                method_name: str = "pca") -> np.ndarray:
        """Fit PCA and transform data."""
        from sklearn.decomposition import PCA
        
        if n_components is None:
            n_components = min(X.shape) - 1
        
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)
        
        self.fitted_transformers[method_name] = pca
        self.embedding_results[method_name] = {
            'embedding': X_transformed,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
        }
        
        return X_transformed
    
    def fit_ica(self, X: np.ndarray, n_components: int = None,
                method_name: str = "ica") -> np.ndarray:
        """Fit ICA and transform data."""
        from sklearn.decomposition import FastICA
        
        if n_components is None:
            n_components = X.shape[1]
        
        ica = FastICA(n_components=n_components, random_state=42)
        X_transformed = ica.fit_transform(X)
        
        self.fitted_transformers[method_name] = ica
        self.embedding_results[method_name] = {
            'embedding': X_transformed,
            'mixing_matrix': ica.mixing_,
            'components': ica.components_
        }
        
        return X_transformed
    
    def fit_tsne(self, X: np.ndarray, n_components: int = 2, perplexity: float = 30.0,
                 method_name: str = "tsne") -> np.ndarray:
        """Fit t-SNE and transform data."""
        from sklearn.manifold import TSNE
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   random_state=42, n_iter=1000)
        X_transformed = tsne.fit_transform(X)
        
        self.fitted_transformers[method_name] = tsne
        self.embedding_results[method_name] = {
            'embedding': X_transformed,
            'kl_divergence': tsne.kl_divergence_,
            'n_iter': tsne.n_iter_
        }
        
        return X_transformed
    
    def fit_isomap(self, X: np.ndarray, n_components: int = 2, n_neighbors: int = 5,
                   method_name: str = "isomap") -> np.ndarray:
        """Fit Isomap and transform data."""
        from sklearn.manifold import Isomap
        
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        X_transformed = isomap.fit_transform(X)
        
        self.fitted_transformers[method_name] = isomap
        self.embedding_results[method_name] = {
            'embedding': X_transformed,
            'reconstruction_error': isomap.reconstruction_error()
        }
        
        return X_transformed
    
    def fit_lle(self, X: np.ndarray, n_components: int = 2, n_neighbors: int = 5,
                method_name: str = "lle") -> np.ndarray:
        """Fit Locally Linear Embedding and transform data."""
        from sklearn.manifold import LocallyLinearEmbedding
        
        lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors,
                                   random_state=42)
        X_transformed = lle.fit_transform(X)
        
        self.fitted_transformers[method_name] = lle
        self.embedding_results[method_name] = {
            'embedding': X_transformed,
            'reconstruction_error': lle.reconstruction_error_
        }
        
        return X_transformed
    
    def fit_nmf(self, X: np.ndarray, n_components: int = 10,
                method_name: str = "nmf") -> np.ndarray:
        """Fit Non-negative Matrix Factorization."""
        from sklearn.decomposition import NMF
        
        # Ensure non-negative data
        X_positive = X - X.min() + 1e-8
        
        nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
        X_transformed = nmf.fit_transform(X_positive)
        
        self.fitted_transformers[method_name] = nmf
        self.embedding_results[method_name] = {
            'embedding': X_transformed,
            'components': nmf.components_,
            'reconstruction_error': nmf.reconstruction_err_
        }
        
        return X_transformed
    
    def compare_methods(self, X: np.ndarray, methods: List[str], 
                       n_components: int = 2) -> Dict:
        """Compare multiple manifold learning methods."""
        results = {}
        
        for method in methods:
            if method == "pca":
                embedding = self.fit_pca(X, n_components, f"compare_{method}")
            elif method == "ica":
                embedding = self.fit_ica(X, n_components, f"compare_{method}")
            elif method == "tsne":
                embedding = self.fit_tsne(X, n_components, method_name=f"compare_{method}")
            elif method == "isomap":
                embedding = self.fit_isomap(X, n_components, method_name=f"compare_{method}")
            elif method == "lle":
                embedding = self.fit_lle(X, n_components, method_name=f"compare_{method}")
            elif method == "nmf":
                embedding = self.fit_nmf(X, n_components, method_name=f"compare_{method}")
            else:
                continue
            
            results[method] = {
                'embedding': embedding,
                'method_info': self.embedding_results[f"compare_{method}"]
            }
        
        return results

class TransferLearningEngine:
    """Transfer learning and domain adaptation engine."""
    
    def __init__(self):
        """Initialize transfer learning engine."""
        self.source_models = {}
        self.target_models = {}
        self.adaptation_results = {}
    
    def feature_extraction_transfer(self, source_model: nn.Module, target_data: torch.Tensor,
                                  target_labels: torch.Tensor, freeze_layers: int = -1) -> nn.Module:
        """Feature extraction transfer learning."""
        
        # Create new model with frozen source features
        target_model = type(source_model)(
            source_model.input_dim, source_model.latent_dim, 
            source_model.hidden_dims
        )
        
        # Copy source model weights
        target_model.load_state_dict(source_model.state_dict())
        
        # Freeze specified layers
        if freeze_layers > 0:
            for i, (name, param) in enumerate(target_model.named_parameters()):
                if i < freeze_layers:
                    param.requires_grad = False
        
        return target_model
    
    def fine_tuning_transfer(self, source_model: nn.Module, target_data: torch.Tensor,
                           target_labels: torch.Tensor, learning_rate: float = 0.001,
                           num_epochs: int = 50) -> Tuple[nn.Module, List]:
        """Fine-tuning transfer learning."""
        
        # Create target model from source
        target_model = type(source_model)(
            source_model.input_dim, source_model.latent_dim,
            source_model.hidden_dims
        )
        target_model.load_state_dict(source_model.state_dict())
        target_model.to(DEVICE)
        
        # Fine-tune with lower learning rate
        optimizer = optim.Adam(target_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        target_data = target_data.to(DEVICE)
        target_labels = target_labels.to(DEVICE)
        
        training_history = []
        
        for epoch in range(num_epochs):
            target_model.train()
            optimizer.zero_grad()
            
            # Forward pass
            recon_x, z = target_model(target_data)
            loss = criterion(recon_x, target_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            training_history.append({
                'epoch': epoch,
                'loss': loss.item()
            })
            
            if epoch % 10 == 0:
                print(f"Fine-tuning Epoch {epoch}: Loss = {loss.item():.6f}")
        
        return target_model, training_history
    
    def domain_adaptation_mmd(self, source_features: torch.Tensor, 
                             target_features: torch.Tensor) -> float:
        """Maximum Mean Discrepancy for domain adaptation."""
        
        def gaussian_kernel(x, y, sigma=1.0):
            """Gaussian RBF kernel."""
            dist = torch.cdist(x, y) ** 2
            return torch.exp(-dist / (2 * sigma ** 2))
        
        # Compute MMD
        xx = gaussian_kernel(source_features, source_features).mean()
        yy = gaussian_kernel(target_features, target_features).mean()
        xy = gaussian_kernel(source_features, target_features).mean()
        
        mmd = xx + yy - 2 * xy
        return mmd.item()
    
    def multitask_learning(self, shared_features: int, task_configs: List[Dict],
                          data_loaders: List[DataLoader]) -> Dict:
        """Multi-task learning with shared representations."""
        
        class MultiTaskModel(nn.Module):
            def __init__(self, input_dim, shared_dim, task_configs):
                super().__init__()
                
                # Shared feature extractor
                self.shared_encoder = nn.Sequential(
                    nn.Linear(input_dim, shared_dim * 2),
                    nn.ReLU(),
                    nn.Linear(shared_dim * 2, shared_dim),
                    nn.ReLU()
                )
                
                # Task-specific heads
                self.task_heads = nn.ModuleList()
                for config in task_configs:
                    head = nn.Sequential(
                        nn.Linear(shared_dim, config['hidden_dim']),
                        nn.ReLU(),
                        nn.Linear(config['hidden_dim'], config['output_dim'])
                    )
                    self.task_heads.append(head)
            
            def forward(self, x, task_id=None):
                shared_features = self.shared_encoder(x)
                
                if task_id is not None:
                    return self.task_heads[task_id](shared_features)
                else:
                    # Return outputs for all tasks
                    outputs = []
                    for head in self.task_heads:
                        outputs.append(head(shared_features))
                    return outputs
        
        # Create multi-task model
        input_dim = task_configs[0]['input_dim']
        model = MultiTaskModel(input_dim, shared_features, task_configs).to(DEVICE)
        
        # Optimizers and criteria
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criteria = [nn.MSELoss() for _ in range(len(task_configs))]
        
        # Training
        training_history = {'total_loss': [], 'task_losses': [[] for _ in range(len(task_configs))]}
        
        for epoch in range(50):
            model.train()
            epoch_losses = [0.0 for _ in range(len(task_configs))]
            
            # Iterate through all tasks
            for task_id, data_loader in enumerate(data_loaders):
                for batch_data, batch_labels in data_loader:
                    batch_data = batch_data.to(DEVICE)
                    batch_labels = batch_labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass for specific task
                    output = model(batch_data, task_id)
                    loss = criteria[task_id](output, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses[task_id] += loss.item()
            
            # Record losses
            total_loss = sum(epoch_losses)
            training_history['total_loss'].append(total_loss)
            for i, loss in enumerate(epoch_losses):
                training_history['task_losses'][i].append(loss)
            
            if epoch % 10 == 0:
                print(f"Multi-task Epoch {epoch}: Total Loss = {total_loss:.6f}")
        
        return {
            'model': model,
            'training_history': training_history
        }

class FeatureLearningEngine:
    """Main engine for feature learning and representation learning."""
    
    def __init__(self):
        """Initialize feature learning engine."""
        self.models = {}
        self.manifold_learner = ManifoldLearner()
        self.transfer_engine = TransferLearningEngine()
        self.training_history = {}
        self.device = DEVICE
    
    def create_autoencoder(self, model_name: str, config: FeatureLearningConfig) -> nn.Module:
        """Create and register autoencoder model."""
        
        if config.method_type == FeatureLearningType.AUTOENCODER:
            model = BaseAutoencoder(
                config.input_dim, config.latent_dim, config.hidden_dims, config.dropout_rate
            )
        elif config.method_type == FeatureLearningType.DENOISING_AUTOENCODER:
            model = DenoisingAutoencoder(
                config.input_dim, config.latent_dim, config.hidden_dims,
                config.noise_factor, config.dropout_rate
            )
        elif config.method_type == FeatureLearningType.SPARSE_AUTOENCODER:
            model = SparseAutoencoder(
                config.input_dim, config.latent_dim, config.hidden_dims,
                config.sparsity_weight, config.dropout_rate
            )
        elif config.method_type == FeatureLearningType.CONTRACTIVE_AUTOENCODER:
            model = ContractiveAutoencoder(
                config.input_dim, config.latent_dim, config.hidden_dims,
                config.contraction_weight, config.dropout_rate
            )
        elif config.method_type == FeatureLearningType.BETA_VAE:
            model = BetaVAE(
                config.input_dim, config.latent_dim, config.hidden_dims,
                config.beta, config.dropout_rate
            )
        elif config.method_type == FeatureLearningType.WAE:
            model = WassersteinAutoencoder(
                config.input_dim, config.latent_dim, config.hidden_dims,
                config.regularization_weight, config.dropout_rate
            )
        else:
            raise ValueError(f"Unknown autoencoder type: {config.method_type}")
        
        model = model.to(self.device)
        self.models[model_name] = model
        return model
    
    def train_autoencoder(self, model_name: str, X_train: torch.Tensor, 
                         config: FeatureLearningConfig, X_val: Optional[torch.Tensor] = None) -> Dict:
        """Train autoencoder model."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        X_train = X_train.to(self.device)
        if X_val is not None:
            X_val = X_val.to(self.device)
        
        training_history = []
        
        for epoch in range(config.num_epochs):
            model.train()
            
            total_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), config.batch_size):
                batch_X = X_train[i:i+config.batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass based on model type
                if isinstance(model, SparseAutoencoder):
                    recon_x, z, sparsity_loss = model(batch_X)
                    recon_loss = F.mse_loss(recon_x, batch_X)
                    loss = recon_loss + sparsity_loss
                    
                elif isinstance(model, ContractiveAutoencoder):
                    recon_x, z, contractive_loss = model(batch_X)
                    recon_loss = F.mse_loss(recon_x, batch_X)
                    loss = recon_loss + contractive_loss
                    
                elif isinstance(model, BetaVAE):
                    recon_x, mu, logvar, z = model(batch_X)
                    recon_loss = F.mse_loss(recon_x, batch_X)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + model.beta * kl_loss / len(batch_X)
                    
                elif isinstance(model, WassersteinAutoencoder):
                    recon_x, z = model(batch_X)
                    recon_loss = F.mse_loss(recon_x, batch_X)
                    
                    # Wasserstein penalty (simplified)
                    z_prior = torch.randn_like(z)
                    wasserstein_penalty = torch.mean((z - z_prior) ** 2)
                    loss = recon_loss + model.regularization_weight * wasserstein_penalty
                    
                else:  # Standard autoencoders
                    recon_x, z = model(batch_X)
                    loss = F.mse_loss(recon_x, batch_X)
                
                # Backward pass
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
                    if isinstance(model, (SparseAutoencoder, ContractiveAutoencoder)):
                        recon_x, _ = model(X_val)[:2]  # Only take first two outputs
                    elif isinstance(model, BetaVAE):
                        recon_x = model(X_val)[0]  # Only reconstruction
                    else:
                        recon_x, _ = model(X_val)
                    
                    val_loss = F.mse_loss(recon_x, X_val).item()
            
            epoch_info = {
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_loss': val_loss
            }
            
            training_history.append(epoch_info)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={avg_loss:.6f}, Val Loss={val_loss:.6f}")
        
        self.training_history[model_name] = training_history
        return training_history
    
    def extract_features(self, model_name: str, X: torch.Tensor) -> torch.Tensor:
        """Extract features using trained autoencoder."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        X = X.to(self.device)
        
        model.eval()
        with torch.no_grad():
            if isinstance(model, BetaVAE):
                mu, logvar = model.encode(X)
                # Use mean of latent distribution
                features = mu
            else:
                features = model.encode(X)
        
        return features
    
    def perform_manifold_learning(self, X: np.ndarray, methods: List[str],
                                n_components: int = 2) -> Dict:
        """Perform manifold learning with multiple methods."""
        return self.manifold_learner.compare_methods(X, methods, n_components)
    
    def evaluate_representation_quality(self, original_data: torch.Tensor,
                                      learned_features: torch.Tensor,
                                      task_labels: Optional[torch.Tensor] = None) -> Dict:
        """Evaluate quality of learned representations."""
        
        # Convert to numpy for sklearn metrics
        features_np = learned_features.detach().cpu().numpy()
        
        results = {}
        
        # Reconstruction error (if autoencoder available)
        if len(self.models) > 0:
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            
            model.eval()
            with torch.no_grad():
                if isinstance(model, BetaVAE):
                    recon_x = model.decode(learned_features)
                else:
                    recon_x = model.decode(learned_features)
                
                recon_error = F.mse_loss(recon_x, original_data).item()
                results['reconstruction_error'] = recon_error
        
        # Intrinsic dimensionality estimate
        if features_np.shape[1] > 1:
            pca = PCA()
            pca.fit(features_np)
            
            # Estimate intrinsic dimensionality (95% variance)
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim = np.argmax(cumsum_var >= 0.95) + 1
            
            results['intrinsic_dimensionality'] = intrinsic_dim
            results['explained_variance_ratio'] = pca.explained_variance_ratio_
        
        # Clustering quality (if we have multiple samples)
        if features_np.shape[0] > 10:
            try:
                from sklearn.cluster import KMeans
                
                # Try different numbers of clusters
                silhouette_scores = []
                k_range = range(2, min(10, features_np.shape[0] // 2))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(features_np)
                    score = silhouette_score(features_np, cluster_labels)
                    silhouette_scores.append(score)
                
                results['best_silhouette_score'] = max(silhouette_scores)
                results['optimal_clusters'] = k_range[np.argmax(silhouette_scores)]
                
            except Exception as e:
                results['clustering_error'] = str(e)
        
        # Task-specific evaluation (if labels provided)
        if task_labels is not None:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_score
                
                # Classification accuracy using learned features
                clf = LogisticRegression(random_state=42, max_iter=1000)
                scores = cross_val_score(clf, features_np, task_labels.cpu().numpy(), cv=5)
                
                results['classification_accuracy_mean'] = scores.mean()
                results['classification_accuracy_std'] = scores.std()
                
            except Exception as e:
                results['classification_error'] = str(e)
        
        return results
    
    def get_model_summary(self, model_name: str) -> Dict:
        """Get model summary and statistics."""
        
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        
        summary = {
            'model_type': type(model).__name__,
            'model_name': model_name,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Model-specific information
        if hasattr(model, 'input_dim'):
            summary['input_dim'] = model.input_dim
        if hasattr(model, 'latent_dim'):
            summary['latent_dim'] = model.latent_dim
        if hasattr(model, 'hidden_dims'):
            summary['hidden_dims'] = model.hidden_dims
        
        # Special parameters for specific models
        if isinstance(model, SparseAutoencoder):
            summary['sparsity_weight'] = model.sparsity_weight
        elif isinstance(model, ContractiveAutoencoder):
            summary['contraction_weight'] = model.contraction_weight
        elif isinstance(model, BetaVAE):
            summary['beta'] = model.beta
        elif isinstance(model, WassersteinAutoencoder):
            summary['regularization_weight'] = model.regularization_weight
        
        # Training history
        if model_name in self.training_history:
            history = self.training_history[model_name]
            if history:
                summary['training_epochs'] = len(history)
                summary['final_train_loss'] = history[-1]['train_loss']
                summary['final_val_loss'] = history[-1]['val_loss']
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    print("Testing Feature Learning and Representation Systems...")
    
    # Generate synthetic data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # High-dimensional data with lower-dimensional structure
    n_samples = 1000
    n_features = 100
    latent_dim = 10
    
    # Generate data with latent structure
    latent_factors = torch.randn(n_samples, latent_dim)
    
    # Create mixing matrix
    mixing_matrix = torch.randn(n_features, latent_dim)
    
    # Generate observed data
    X = torch.matmul(latent_factors, mixing_matrix.T)
    X += 0.1 * torch.randn_like(X)  # Add noise
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    
    # Initialize engine
    engine = FeatureLearningEngine()
    
    # Test different autoencoder types
    autoencoder_configs = {
        "standard_ae": FeatureLearningConfig(
            method_type=FeatureLearningType.AUTOENCODER,
            input_dim=n_features,
            latent_dim=20,
            hidden_dims=[64, 32],
            learning_rate=0.001,
            num_epochs=50,
            batch_size=32,
            dropout_rate=0.1,
            regularization_weight=0.0,
            noise_factor=0.0,
            sparsity_weight=0.0,
            beta=1.0,
            contraction_weight=0.0
        ),
        "denoising_ae": FeatureLearningConfig(
            method_type=FeatureLearningType.DENOISING_AUTOENCODER,
            input_dim=n_features,
            latent_dim=20,
            hidden_dims=[64, 32],
            learning_rate=0.001,
            num_epochs=50,
            batch_size=32,
            dropout_rate=0.1,
            regularization_weight=0.0,
            noise_factor=0.2,
            sparsity_weight=0.0,
            beta=1.0,
            contraction_weight=0.0
        ),
        "sparse_ae": FeatureLearningConfig(
            method_type=FeatureLearningType.SPARSE_AUTOENCODER,
            input_dim=n_features,
            latent_dim=20,
            hidden_dims=[64, 32],
            learning_rate=0.001,
            num_epochs=50,
            batch_size=32,
            dropout_rate=0.1,
            regularization_weight=0.0,
            noise_factor=0.0,
            sparsity_weight=0.01,
            beta=1.0,
            contraction_weight=0.0
        ),
        "beta_vae": FeatureLearningConfig(
            method_type=FeatureLearningType.BETA_VAE,
            input_dim=n_features,
            latent_dim=20,
            hidden_dims=[64, 32],
            learning_rate=0.001,
            num_epochs=50,
            batch_size=32,
            dropout_rate=0.1,
            regularization_weight=0.0,
            noise_factor=0.0,
            sparsity_weight=0.0,
            beta=4.0,
            contraction_weight=0.0
        )
    }
    
    # Train and evaluate autoencoders
    print("\n=== Training Autoencoders ===")
    
    trained_models = {}
    for model_name, config in autoencoder_configs.items():
        print(f"\nTraining {model_name}...")
        
        # Create and train model
        model = engine.create_autoencoder(model_name, config)
        history = engine.train_autoencoder(model_name, X_train, config, X_test)
        
        # Extract features
        features = engine.extract_features(model_name, X_test)
        
        # Evaluate representation quality
        evaluation = engine.evaluate_representation_quality(
            X_test, features
        )
        
        trained_models[model_name] = {
            'model': model,
            'history': history,
            'features': features,
            'evaluation': evaluation
        }
        
        print(f"Final train loss: {history[-1]['train_loss']:.6f}")
        print(f"Final val loss: {history[-1]['val_loss']:.6f}")
        print(f"Reconstruction error: {evaluation.get('reconstruction_error', 'N/A')}")
        if 'intrinsic_dimensionality' in evaluation:
            print(f"Estimated intrinsic dimensionality: {evaluation['intrinsic_dimensionality']}")
    
    # Test manifold learning
    print("\n=== Testing Manifold Learning ===")
    
    # Use features from standard autoencoder for manifold learning
    ae_features = trained_models['standard_ae']['features'].detach().cpu().numpy()
    
    # Apply multiple manifold learning methods
    methods = ['pca', 'ica', 'tsne', 'isomap', 'lle']
    manifold_results = engine.perform_manifold_learning(ae_features, methods, n_components=2)
    
    for method, result in manifold_results.items():
        embedding = result['embedding']
        print(f"\n{method.upper()}:")
        print(f"  Embedding shape: {embedding.shape}")
        
        if method == 'pca':
            var_ratio = result['method_info']['explained_variance_ratio']
            print(f"  Explained variance (first 2 components): {var_ratio[:2].sum():.4f}")
        elif method == 'tsne':
            kl_div = result['method_info']['kl_divergence']
            print(f"  Final KL divergence: {kl_div:.4f}")
        elif method in ['isomap', 'lle']:
            recon_error = result['method_info']['reconstruction_error']
            print(f"  Reconstruction error: {recon_error:.6f}")
    
    # Test transfer learning
    print("\n=== Testing Transfer Learning ===")
    
    # Create a "source" dataset (similar but different)
    X_source = X_train + 0.5 * torch.randn_like(X_train)
    X_target = X_test + 0.3 * torch.randn_like(X_test)
    
    # Use standard autoencoder as source model
    source_model = trained_models['standard_ae']['model']
    
    # Feature extraction transfer
    print("Performing feature extraction transfer...")
    target_model_fe = engine.transfer_engine.feature_extraction_transfer(
        source_model, X_target, None, freeze_layers=2
    )
    
    # Fine-tuning transfer
    print("Performing fine-tuning transfer...")
    target_model_ft, ft_history = engine.transfer_engine.fine_tuning_transfer(
        source_model, X_target, None, learning_rate=0.0001, num_epochs=20
    )
    
    print(f"Fine-tuning final loss: {ft_history[-1]['loss']:.6f}")
    
    # Domain adaptation with MMD
    source_features = engine.extract_features('standard_ae', X_source)
    target_features = engine.extract_features('standard_ae', X_target)
    
    mmd_distance = engine.transfer_engine.domain_adaptation_mmd(
        source_features, target_features
    )
    print(f"MMD distance between domains: {mmd_distance:.6f}")
    
    # Model summaries
    print("\n=== Model Summaries ===")
    for model_name in autoencoder_configs.keys():
        summary = engine.get_model_summary(model_name)
        print(f"\n{model_name}:")
        for key, value in summary.items():
            if key not in ['model_name', 'hidden_dims']:
                print(f"  {key}: {value}")
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    print(f"{'Model':<15} {'Train Loss':<12} {'Val Loss':<12} {'Recon Error':<12} {'Intrinsic Dim':<12}")
    print("-" * 65)
    
    for model_name, results in trained_models.items():
        history = results['history']
        evaluation = results['evaluation']
        
        train_loss = history[-1]['train_loss']
        val_loss = history[-1]['val_loss']
        recon_error = evaluation.get('reconstruction_error', 0.0)
        intrinsic_dim = evaluation.get('intrinsic_dimensionality', 0)
        
        print(f"{model_name:<15} {train_loss:<12.6f} {val_loss:<12.6f} "
              f"{recon_error:<12.6f} {intrinsic_dim:<12}")
    
    print("\nFeature learning and representation systems testing completed successfully!")