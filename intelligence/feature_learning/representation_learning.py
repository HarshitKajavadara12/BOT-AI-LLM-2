"""
Representation Learning for Financial Data
Advanced techniques for learning meaningful representations from market data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from abc import ABC, abstractmethod
import math
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
except ImportError:
    UMAP = None
import warnings


class RepresentationLearner(ABC, nn.Module):
    """Abstract base class for representation learning methods"""
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to representation"""
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode representation to reconstruction"""
        pass
    
    @abstractmethod
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute representation learning loss"""
        pass


_ACTIVATION_MAP = {
    'relu': 'ReLU',
    'elu': 'ELU',
    'leaky_relu': 'LeakyReLU',
    'gelu': 'GELU',
    'tanh': 'Tanh',
    'sigmoid': 'Sigmoid',
    'selu': 'SELU',
}


class AutoEncoder(RepresentationLearner):
    """
    Standard autoencoder for dimensionality reduction
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [256, 128],
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        encoder_layers = []
        
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            if i < len(encoder_dims) - 2:  # No activation on last layer
                act_name = _ACTIVATION_MAP.get(activation, activation)
                encoder_layers.append(getattr(nn, act_name)())
                encoder_layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        decoder_layers = []
        
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2:  # No activation on last layer
                act_name = _ACTIVATION_MAP.get(activation, activation)
                decoder_layers.append(getattr(nn, act_name)())
                decoder_layers.append(nn.Dropout(dropout))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')
        return {
            'total_loss': mse_loss,
            'reconstruction_loss': mse_loss,
            'regularization_loss': torch.tensor(0.0, device=x.device)
        }


class SparseAutoEncoder(RepresentationLearner):
    """
    Sparse autoencoder with L1 regularization on hidden activations
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [256, 128],
        sparsity_weight: float = 1e-3,
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.sparsity_weight = sparsity_weight
        
        # Build autoencoder
        self.autoencoder = AutoEncoder(
            input_dim, latent_dim, hidden_dims, activation, dropout
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder(x)
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Reconstruction loss
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # Sparsity regularization (L1 on latent codes)
        sparsity_loss = torch.mean(torch.abs(z))
        
        total_loss = mse_loss + self.sparsity_weight * sparsity_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': mse_loss,
            'regularization_loss': sparsity_loss
        }


class ContrastiveAutoEncoder(RepresentationLearner):
    """
    Contrastive autoencoder for learning discriminative representations
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [256, 128],
        temperature: float = 0.1,
        contrastive_weight: float = 1.0
    ):
        super().__init__()
        
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        
        # Build autoencoder
        self.autoencoder = AutoEncoder(input_dim, latent_dim, hidden_dims)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        h = self.projection_head(z)  # For contrastive learning
        return x_recon, z, h
    
    def contrastive_loss(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE-style)
        Assumes that consecutive samples are positive pairs
        """
        batch_size = h.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=h.device)
        
        # Normalize representations
        h_norm = F.normalize(h, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(h_norm, h_norm.t()) / self.temperature
        
        # Create positive pairs (consecutive samples)
        labels = torch.arange(batch_size, device=h.device)
        # Shift labels to create pairs
        positive_labels = (labels + 1) % batch_size
        
        # Contrastive loss
        exp_sim = torch.exp(sim_matrix)
        pos_sim = exp_sim[labels, positive_labels]
        neg_sim = exp_sim.sum(dim=1) - torch.diag(exp_sim)
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        
        return loss
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor, h: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Reconstruction loss
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # Contrastive loss
        if h is not None:
            contrastive_loss = self.contrastive_loss(h)
        else:
            contrastive_loss = torch.tensor(0.0, device=x.device)
        
        total_loss = mse_loss + self.contrastive_weight * contrastive_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': mse_loss,
            'regularization_loss': contrastive_loss
        }


class TemporalAutoEncoder(RepresentationLearner):
    """
    Temporal autoencoder for time series representation learning
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        
        # Encoder output dimension
        encoder_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Latent projection
        self.encoder_proj = nn.Linear(encoder_output_dim, latent_dim)
        
        # Decoder
        self.decoder_proj = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, input_dim, num_layers,
            batch_first=True
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time series to latent representation
        
        Args:
            x: Input time series [batch_size, seq_len, input_dim]
        
        Returns:
            z: Latent representation [batch_size, latent_dim]
        """
        # LSTM encoding
        lstm_out, (hidden, cell) = self.encoder_lstm(x)
        
        # Use final hidden state (concatenate if bidirectional)
        if self.encoder_lstm.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            final_hidden = hidden[-1]
        
        # Project to latent space
        z = self.encoder_proj(final_hidden)
        
        return z
    
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation to time series
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            seq_len: Length of output sequence
        
        Returns:
            x_recon: Reconstructed time series [batch_size, seq_len, input_dim]
        """
        batch_size = z.size(0)
        
        # Project to decoder input
        decoder_input = self.decoder_proj(z)
        
        # Repeat for each time step
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)
        
        # LSTM decoding
        x_recon, _ = self.decoder_lstm(decoder_input)
        
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        z = self.encode(x)
        x_recon = self.decode(z, seq_len)
        return x_recon, z
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')
        return {
            'total_loss': mse_loss,
            'reconstruction_loss': mse_loss,
            'regularization_loss': torch.tensor(0.0, device=x.device)
        }


class FactorAutoEncoder(RepresentationLearner):
    """
    Factor autoencoder for learning financial factor models
    """
    
    def __init__(
        self,
        n_assets: int,
        n_factors: int,
        hidden_dims: List[int] = [64, 32],
        orthogonal_factors: bool = True,
        factor_sparsity: float = 0.01
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.orthogonal_factors = orthogonal_factors
        self.factor_sparsity = factor_sparsity
        
        # Factor loadings (decoder)
        self.factor_loadings = nn.Parameter(torch.randn(n_assets, n_factors) * 0.1)
        
        # Factor encoder
        encoder_dims = [n_assets] + hidden_dims + [n_factors]
        encoder_layers = []
        
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            if i < len(encoder_dims) - 2:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(0.1))
        
        self.factor_encoder = nn.Sequential(*encoder_layers)
        
        # Idiosyncratic risk parameters
        self.log_idiosyncratic_var = nn.Parameter(torch.zeros(n_assets))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract factors from asset returns"""
        return self.factor_encoder(x)
    
    def decode(self, factors: torch.Tensor) -> torch.Tensor:
        """Reconstruct asset returns from factors"""
        # Apply orthogonality constraint if specified
        if self.orthogonal_factors and self.training:
            loadings = self._orthogonalize_loadings()
        else:
            loadings = self.factor_loadings
        
        # Factor model: returns = loadings @ factors + idiosyncratic
        return torch.mm(factors, loadings.t())
    
    def _orthogonalize_loadings(self) -> torch.Tensor:
        """Apply Gram-Schmidt orthogonalization to factor loadings"""
        loadings = self.factor_loadings
        
        # QR decomposition for orthogonalization
        Q, R = torch.qr(loadings)
        
        # Ensure positive diagonal of R
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        
        return Q
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        factors = self.encode(x)
        x_recon = self.decode(factors)
        return x_recon, factors
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, factors: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Reconstruction loss with heteroskedastic noise
        idiosyncratic_var = torch.exp(self.log_idiosyncratic_var)
        residuals = x - x_recon
        
        # Weighted MSE loss (inverse variance weighting)
        weighted_mse = torch.mean((residuals**2) / idiosyncratic_var.unsqueeze(0))
        
        # Log-likelihood of idiosyncratic variances
        log_det_loss = torch.sum(self.log_idiosyncratic_var)
        
        # Factor sparsity regularization
        sparsity_loss = torch.mean(torch.abs(factors))
        
        # Orthogonality regularization
        if self.orthogonal_factors:
            factor_cov = torch.mm(factors.t(), factors) / factors.size(0)
            identity = torch.eye(self.n_factors, device=factors.device)
            orthogonal_loss = torch.norm(factor_cov - identity, 'fro')**2
        else:
            orthogonal_loss = torch.tensor(0.0, device=x.device)
        
        total_loss = (weighted_mse + log_det_loss + 
                     self.factor_sparsity * sparsity_loss + 
                     0.01 * orthogonal_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': weighted_mse,
            'regularization_loss': sparsity_loss + orthogonal_loss
        }
    
    def get_factor_loadings(self) -> torch.Tensor:
        """Get factor loadings matrix"""
        if self.orthogonal_factors:
            return self._orthogonalize_loadings()
        return self.factor_loadings
    
    def get_idiosyncratic_risks(self) -> torch.Tensor:
        """Get idiosyncratic risk estimates"""
        return torch.exp(self.log_idiosyncratic_var)


class NonlinearPCA(nn.Module):
    """
    Nonlinear Principal Component Analysis using autoencoders
    """
    
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'tanh'
    ):
        super().__init__()
        
        self.n_components = n_components
        
        # Use autoencoder architecture
        self.autoencoder = AutoEncoder(
            input_dim=input_dim,
            latent_dim=n_components,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=0.0  # No dropout for PCA
        )
    
    def fit(
        self,
        X: torch.Tensor,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64
    ):
        """Fit nonlinear PCA"""
        
        # Create data loader
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Training loop
        self.train()
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            for batch in dataloader:
                x_batch = batch[0]
                
                optimizer.zero_grad()
                
                # Forward pass
                x_recon, z = self.autoencoder(x_batch)
                
                # Compute loss
                loss_dict = self.autoencoder.compute_loss(x_batch, x_recon, z)
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform data to principal components"""
        self.eval()
        with torch.no_grad():
            return self.autoencoder.encode(X)
    
    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """Transform principal components back to original space"""
        self.eval()
        with torch.no_grad():
            return self.autoencoder.decode(Z)
    
    def fit_transform(self, X: torch.Tensor, **fit_params) -> torch.Tensor:
        """Fit model and transform data"""
        self.fit(X, **fit_params)
        return self.transform(X)


def train_representation_learner(
    model: RepresentationLearner,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: torch.device = None
) -> Dict[str, List[float]]:
    """
    Train representation learning model
    
    Args:
        model: Representation learning model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        lr: Learning rate
        device: Training device
    
    Returns:
        Training history
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_reg_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_reg_loss': []
    }
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_recon_losses = []
        train_reg_losses = []
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, ContrastiveAutoEncoder):
                x_recon, z, h = model(x)
                loss_dict = model.compute_loss(x, x_recon, z, h)
            else:
                x_recon, z = model(x)
                loss_dict = model.compute_loss(x, x_recon, z)
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_recon_losses.append(loss_dict['reconstruction_loss'].item())
            train_reg_losses.append(loss_dict['regularization_loss'].item())
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_losses = []
            val_recon_losses = []
            val_reg_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0].to(device)
                    else:
                        x = batch.to(device)
                    
                    # Forward pass
                    if isinstance(model, ContrastiveAutoEncoder):
                        x_recon, z, h = model(x)
                        loss_dict = model.compute_loss(x, x_recon, z, h)
                    else:
                        x_recon, z = model(x)
                        loss_dict = model.compute_loss(x, x_recon, z)
                    
                    val_losses.append(loss_dict['total_loss'].item())
                    val_recon_losses.append(loss_dict['reconstruction_loss'].item())
                    val_reg_losses.append(loss_dict['regularization_loss'].item())
        
        # Update history
        history['train_loss'].append(np.mean(train_losses))
        history['train_recon_loss'].append(np.mean(train_recon_losses))
        history['train_reg_loss'].append(np.mean(train_reg_losses))
        
        if val_loader is not None:
            history['val_loss'].append(np.mean(val_losses))
            history['val_recon_loss'].append(np.mean(val_recon_losses))
            history['val_reg_loss'].append(np.mean(val_reg_losses))
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {history['train_loss'][-1]:.6f}")
            if val_loader is not None:
                print(f"  Val Loss: {history['val_loss'][-1]:.6f}")
    
    return history


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    
    print("Testing Representation Learning Models...")
    
    # Generate synthetic financial data
    n_samples = 1000
    n_features = 50
    n_factors = 5
    
    # Create factor model data
    true_factors = torch.randn(n_samples, n_factors)
    true_loadings = torch.randn(n_features, n_factors) * 0.3
    noise = torch.randn(n_samples, n_features) * 0.1
    
    X = torch.mm(true_factors, true_loadings.t()) + noise
    
    # Test different representation learning models
    models = {
        'autoencoder': AutoEncoder(n_features, n_factors, [32, 16]),
        'sparse_ae': SparseAutoEncoder(n_features, n_factors, [32, 16], sparsity_weight=0.01),
        'factor_ae': FactorAutoEncoder(n_features, n_factors, [32, 16])
    }
    
    # Create data loader
    dataset = TensorDataset(X)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Train model
        history = train_representation_learner(
            model, train_loader, n_epochs=50, lr=1e-3
        )
        
        # Test encoding/decoding
        model.eval()
        with torch.no_grad():
            test_sample = X[:10]
            
            if isinstance(model, FactorAutoEncoder):
                x_recon, factors = model(test_sample)
                print(f"  Reconstruction error: {F.mse_loss(x_recon, test_sample).item():.6f}")
                print(f"  Factor loadings shape: {model.get_factor_loadings().shape}")
                print(f"  Idiosyncratic risks mean: {model.get_idiosyncratic_risks().mean().item():.6f}")
            else:
                x_recon, z = model(test_sample)
                print(f"  Reconstruction error: {F.mse_loss(x_recon, test_sample).item():.6f}")
                print(f"  Latent representation shape: {z.shape}")
    
    # Test temporal autoencoder
    print("\nTesting Temporal AutoEncoder...")
    
    # Generate time series data
    seq_len = 20
    n_series = 100
    input_dim = 10
    
    time_series = torch.randn(n_series, seq_len, input_dim)
    
    temporal_ae = TemporalAutoEncoder(input_dim, latent_dim=5, hidden_dim=16)
    
    # Test forward pass
    x_recon, z = temporal_ae(time_series[:5])
    print(f"Temporal AE reconstruction shape: {x_recon.shape}")
    print(f"Temporal AE latent shape: {z.shape}")
    
    # Test nonlinear PCA
    print("\nTesting Nonlinear PCA...")
    
    nlpca = NonlinearPCA(input_dim=n_features, n_components=n_factors, hidden_dims=[32])
    
    # Fit and transform
    Z_transformed = nlpca.fit_transform(X, n_epochs=50, lr=1e-3)
    X_reconstructed = nlpca.inverse_transform(Z_transformed)
    
    reconstruction_error = F.mse_loss(X_reconstructed, X)
    print(f"Nonlinear PCA reconstruction error: {reconstruction_error.item():.6f}")
    print(f"Transformed data shape: {Z_transformed.shape}")
    
    print("\nDone!")