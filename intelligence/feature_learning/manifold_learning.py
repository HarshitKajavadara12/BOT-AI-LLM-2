"""
Manifold Learning for Financial Data
Advanced dimensionality reduction and manifold discovery techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from abc import ABC, abstractmethod
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigvals
import umap


class ManifoldLearner(ABC):
    """Abstract base class for manifold learning methods"""
    
    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Learn manifold and transform data
        
        Args:
            X: Input data [n_samples, n_features]
        
        Returns:
            Transformed data [n_samples, n_components]
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using learned manifold
        
        Args:
            X: New data to transform
        
        Returns:
            Transformed data
        """
        pass
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform from manifold space (if supported)
        
        Args:
            X_transformed: Data in manifold space
        
        Returns:
            Reconstructed data in original space
        """
        raise NotImplementedError("Inverse transform not supported by this method")


class VariationalAutoencoder(nn.Module, ManifoldLearner):
    """
    Variational Autoencoder for nonlinear manifold learning
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dims: List[int] = [128, 64],
        beta: float = 1.0,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        decoder_layers.extend([
            nn.Linear(latent_dim, hidden_dims[-1]),
            self.activation,
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.Dropout(0.2)
        ])
        
        prev_dim = hidden_dims[-1]
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # For manifold learning interface
        self.fitted = False
        self.scaler = StandardScaler()
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE loss function"""
        
        # Reconstruction loss
        recon_loss = nn.MSELoss(reduction='sum')(reconstruction, x)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def fit_transform(self, X: np.ndarray, epochs: int = 100, batch_size: int = 64) -> np.ndarray:
        """Fit VAE and transform data"""
        
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (data,) in enumerate(dataloader):
                optimizer.zero_grad()
                
                reconstruction, mu, logvar = self(data)
                loss, recon_loss, kl_loss = self.loss_function(reconstruction, data, mu, logvar)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss:.4f}')
        
        self.fitted = True
        
        # Transform data
        return self.transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to latent space"""
        if not self.fitted:
            raise ValueError("Model must be fitted before transform")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(X_tensor)
            return mu.numpy()
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform from latent space"""
        if not self.fitted:
            raise ValueError("Model must be fitted before inverse transform")
        
        Z_tensor = torch.FloatTensor(X_transformed)
        
        self.eval()
        with torch.no_grad():
            reconstruction = self.decode(Z_tensor)
            reconstruction_scaled = reconstruction.numpy()
            
        # Inverse standardize
        return self.scaler.inverse_transform(reconstruction_scaled)


class DiffusionMaps(ManifoldLearner):
    """
    Diffusion Maps for manifold learning
    Based on diffusion processes on data manifolds
    """
    
    def __init__(
        self,
        n_components: int = 2,
        epsilon: Optional[float] = None,
        alpha: float = 1.0,
        k: int = 10
    ):
        self.n_components = n_components
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k
        
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.fitted_data_ = None
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute Gaussian kernel matrix"""
        
        # Compute pairwise distances
        distances = pairwise_distances(X, metric='euclidean')
        
        # Auto-select epsilon if not provided
        if self.epsilon is None:
            # Use median of k-nearest neighbor distances
            nn = NearestNeighbors(n_neighbors=self.k)
            nn.fit(X)
            knn_distances, _ = nn.kneighbors(X)
            self.epsilon = np.median(knn_distances[:, -1])
        
        # Gaussian kernel
        K = np.exp(-distances**2 / (2 * self.epsilon**2))
        
        return K
    
    def _normalize_kernel(self, K: np.ndarray) -> np.ndarray:
        """Apply diffusion maps normalization"""
        
        # Degree normalization
        d = np.sum(K, axis=1)
        d_alpha = np.power(d, self.alpha)
        
        # Normalize by degree
        D_alpha_inv = np.diag(1.0 / d_alpha)
        K_normalized = D_alpha_inv @ K @ D_alpha_inv
        
        # Row-normalize to get transition matrix
        row_sums = np.sum(K_normalized, axis=1)
        P = K_normalized / row_sums[:, np.newaxis]
        
        return P
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit diffusion maps and transform data"""
        
        self.fitted_data_ = X.copy()
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)
        
        # Normalize kernel
        P = self._normalize_kernel(K)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        
        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        
        # Transform data (skip first eigenvector, which is constant)
        n_components = min(self.n_components, len(eigenvalues) - 1)
        embedding = eigenvectors[:, 1:n_components+1] * eigenvalues[1:n_components+1]
        
        return embedding
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using Nyström extension"""
        if self.fitted_data_ is None:
            raise ValueError("Model must be fitted before transform")
        
        # Compute kernel between new data and fitted data
        distances = pairwise_distances(X, self.fitted_data_, metric='euclidean')
        K_new = np.exp(-distances**2 / (2 * self.epsilon**2))
        
        # Normalize
        d_fitted = np.sum(np.exp(-pairwise_distances(self.fitted_data_)**2 / (2 * self.epsilon**2)), axis=1)
        d_new = np.sum(K_new, axis=1)
        
        d_alpha_fitted = np.power(d_fitted, self.alpha)
        d_alpha_new = np.power(d_new, self.alpha)
        
        # Apply normalization
        K_normalized = K_new / (d_alpha_new[:, np.newaxis] * d_alpha_fitted[np.newaxis, :])
        
        # Project onto eigenvectors
        n_components = min(self.n_components, len(self.eigenvalues_) - 1)
        embedding = (K_normalized @ self.eigenvectors_[:, 1:n_components+1]) / self.eigenvalues_[1:n_components+1]
        
        return embedding


class LaplacianEigenmaps(ManifoldLearner):
    """
    Laplacian Eigenmaps for manifold learning
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        affinity: str = 'knn',  # 'knn' or 'rbf'
        gamma: Optional[float] = None
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.gamma = gamma
        
        self.embedding_ = None
        self.affinity_matrix_ = None
        self.fitted_data_ = None
    
    def _compute_affinity_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute affinity matrix"""
        
        n_samples = X.shape[0]
        W = np.zeros((n_samples, n_samples))
        
        if self.affinity == 'knn':
            # k-nearest neighbors graph
            nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
            nn.fit(X)
            distances, indices = nn.kneighbors(X)
            
            for i in range(n_samples):
                for j, neighbor_idx in enumerate(indices[i, 1:]):  # Skip self
                    W[i, neighbor_idx] = 1.0
                    W[neighbor_idx, i] = 1.0  # Make symmetric
        
        elif self.affinity == 'rbf':
            # RBF kernel
            if self.gamma is None:
                distances = pairwise_distances(X)
                self.gamma = 1.0 / (2 * np.median(distances)**2)
            
            distances = pairwise_distances(X, metric='euclidean')
            W = np.exp(-self.gamma * distances**2)
            np.fill_diagonal(W, 0)  # Remove self-connections
        
        return W
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit Laplacian eigenmaps and transform data"""
        
        self.fitted_data_ = X.copy()
        
        # Compute affinity matrix
        W = self._compute_affinity_matrix(X)
        self.affinity_matrix_ = W
        
        # Compute Laplacian matrix
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        
        # Generalized eigenvalue problem: L v = λ D v
        try:
            eigenvalues, eigenvectors = scipy.linalg.eigh(L, D)
        except:
            # Fallback to regular eigenvalue problem with normalized Laplacian
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-10))
            L_norm = D_inv_sqrt @ L @ D_inv_sqrt
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        
        # Sort by eigenvalue (ascending for Laplacian)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Use smallest non-zero eigenvalues (skip first if close to zero)
        start_idx = 1 if abs(eigenvalues[0]) < 1e-10 else 0
        end_idx = start_idx + self.n_components
        
        self.embedding_ = eigenvectors[:, start_idx:end_idx]
        
        return self.embedding_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data (limited support)"""
        # Laplacian eigenmaps doesn't have natural out-of-sample extension
        # This is a simplified approximation
        if self.fitted_data_ is None:
            raise ValueError("Model must be fitted before transform")
        
        # Find nearest neighbors in fitted data
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(self.fitted_data_)
        distances, indices = nn.kneighbors(X)
        
        # Weighted average of neighbor embeddings
        X_transformed = np.zeros((X.shape[0], self.n_components))
        
        for i in range(X.shape[0]):
            weights = 1.0 / (distances[i] + 1e-10)
            weights /= np.sum(weights)
            
            for j, neighbor_idx in enumerate(indices[i]):
                X_transformed[i] += weights[j] * self.embedding_[neighbor_idx]
        
        return X_transformed


class ManifoldLearningEnsemble:
    """
    Ensemble of manifold learning methods
    """
    
    def __init__(
        self,
        methods: Optional[Dict[str, ManifoldLearner]] = None,
        n_components: int = 2,
        combine_method: str = 'average'  # 'average', 'weighted', 'stacking'
    ):
        if methods is None:
            methods = {
                'PCA': self._create_pca(n_components),
                'UMAP': self._create_umap(n_components),
                'tSNE': self._create_tsne(n_components),
                'Isomap': self._create_isomap(n_components),
                'Diffusion': DiffusionMaps(n_components=n_components)
            }
        
        self.methods = methods
        self.n_components = n_components
        self.combine_method = combine_method
        self.embeddings_ = {}
        self.combined_embedding_ = None
    
    def _create_pca(self, n_components: int):
        """Create PCA wrapper"""
        class PCAWrapper(ManifoldLearner):
            def __init__(self, n_components):
                self.pca = PCA(n_components=n_components)
                self.fitted = False
            
            def fit_transform(self, X):
                self.fitted = True
                return self.pca.fit_transform(X)
            
            def transform(self, X):
                if not self.fitted:
                    raise ValueError("Must fit before transform")
                return self.pca.transform(X)
            
            def inverse_transform(self, X_transformed):
                return self.pca.inverse_transform(X_transformed)
        
        return PCAWrapper(n_components)
    
    def _create_umap(self, n_components: int):
        """Create UMAP wrapper"""
        class UMAPWrapper(ManifoldLearner):
            def __init__(self, n_components):
                self.umap = umap.UMAP(n_components=n_components, random_state=42)
                self.fitted = False
            
            def fit_transform(self, X):
                self.fitted = True
                return self.umap.fit_transform(X)
            
            def transform(self, X):
                if not self.fitted:
                    raise ValueError("Must fit before transform")
                return self.umap.transform(X)
        
        return UMAPWrapper(n_components)
    
    def _create_tsne(self, n_components: int):
        """Create t-SNE wrapper"""
        class TSNEWrapper(ManifoldLearner):
            def __init__(self, n_components):
                self.tsne = TSNE(n_components=n_components, random_state=42)
                self.fitted = False
                self.fitted_data = None
            
            def fit_transform(self, X):
                self.fitted = True
                self.fitted_data = X
                return self.tsne.fit_transform(X)
            
            def transform(self, X):
                # t-SNE doesn't support out-of-sample, return fitted data embedding
                if not self.fitted:
                    raise ValueError("Must fit before transform")
                # For new data, re-run t-SNE (not ideal but necessary)
                combined_data = np.vstack([self.fitted_data, X])
                embedding = TSNE(n_components=self.tsne.n_components, random_state=42).fit_transform(combined_data)
                return embedding[len(self.fitted_data):]
        
        return TSNEWrapper(n_components)
    
    def _create_isomap(self, n_components: int):
        """Create Isomap wrapper"""
        class IsomapWrapper(ManifoldLearner):
            def __init__(self, n_components):
                self.isomap = Isomap(n_components=n_components)
                self.fitted = False
            
            def fit_transform(self, X):
                self.fitted = True
                return self.isomap.fit_transform(X)
            
            def transform(self, X):
                if not self.fitted:
                    raise ValueError("Must fit before transform")
                return self.isomap.transform(X)
        
        return IsomapWrapper(n_components)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit all methods and combine embeddings"""
        
        # Standardize input data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit each method
        for name, method in self.methods.items():
            try:
                print(f"Fitting {name}...")
                embedding = method.fit_transform(X_scaled)
                self.embeddings_[name] = embedding
            except Exception as e:
                print(f"Warning: {name} failed with error: {e}")
                continue
        
        # Combine embeddings
        if self.combine_method == 'average':
            # Simple average
            embeddings_array = np.stack(list(self.embeddings_.values()), axis=0)
            self.combined_embedding_ = np.mean(embeddings_array, axis=0)
        
        elif self.combine_method == 'weighted':
            # Weight by explained variance (for methods that support it)
            weights = []
            embeddings_list = []
            
            for name, embedding in self.embeddings_.items():
                if name == 'PCA':
                    weight = np.sum(self.methods[name].pca.explained_variance_ratio_)
                else:
                    # Equal weight for methods without explained variance
                    weight = 1.0
                
                weights.append(weight)
                embeddings_list.append(embedding)
            
            weights = np.array(weights)
            weights /= np.sum(weights)
            
            self.combined_embedding_ = np.zeros_like(embeddings_list[0])
            for i, embedding in enumerate(embeddings_list):
                self.combined_embedding_ += weights[i] * embedding
        
        return self.combined_embedding_
    
    def plot_embeddings(self, X: np.ndarray, labels: Optional[np.ndarray] = None):
        """Plot all embeddings"""
        
        n_methods = len(self.embeddings_)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, embedding) in enumerate(self.embeddings_.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if labels is not None:
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', alpha=0.7)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
            
            ax.set_title(f'{name} Embedding')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        # Hide empty subplots
        for i in range(n_methods, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.show()


def evaluate_manifold_quality(
    X_original: np.ndarray,
    X_embedded: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate quality of manifold embedding
    
    Args:
        X_original: Original high-dimensional data
        X_embedded: Low-dimensional embedding
        k: Number of neighbors for evaluation
    
    Returns:
        Quality metrics
    """
    
    # Trustworthiness: measures if close points in embedding are close in original space
    nn_original = NearestNeighbors(n_neighbors=k+1)
    nn_original.fit(X_original)
    _, indices_original = nn_original.kneighbors(X_original)
    
    nn_embedded = NearestNeighbors(n_neighbors=k+1)
    nn_embedded.fit(X_embedded)
    _, indices_embedded = nn_embedded.kneighbors(X_embedded)
    
    n_samples = X_original.shape[0]
    
    # Trustworthiness
    trustworthiness = 0
    for i in range(n_samples):
        neighbors_original = set(indices_original[i, 1:])  # Skip self
        neighbors_embedded = set(indices_embedded[i, 1:])  # Skip self
        intersection = neighbors_original & neighbors_embedded
        trustworthiness += len(intersection) / k
    
    trustworthiness /= n_samples
    
    # Continuity: measures if close points in original space are close in embedding
    continuity = 0
    for i in range(n_samples):
        neighbors_original = set(indices_original[i, 1:])
        neighbors_embedded = set(indices_embedded[i, 1:])
        intersection = neighbors_original & neighbors_embedded
        continuity += len(intersection) / k
    
    continuity /= n_samples
    
    # Stress (normalized)
    distances_original = pairwise_distances(X_original)
    distances_embedded = pairwise_distances(X_embedded)
    
    stress = np.sum((distances_original - distances_embedded)**2) / np.sum(distances_original**2)
    
    return {
        'trustworthiness': trustworthiness,
        'continuity': continuity,
        'stress': stress,
        'neighborhood_preservation': (trustworthiness + continuity) / 2
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    print("Testing Manifold Learning Methods...")
    
    # Generate synthetic manifold data (Swiss roll)
    from sklearn.datasets import make_swiss_roll
    
    n_samples = 1000
    X, color = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    
    print(f"Original data shape: {X.shape}")
    
    # Test individual methods
    methods = {
        'VAE': VariationalAutoencoder(input_dim=X.shape[1], latent_dim=2, hidden_dims=[32, 16]),
        'Diffusion Maps': DiffusionMaps(n_components=2, k=10),
        'Laplacian Eigenmaps': LaplacianEigenmaps(n_components=2, n_neighbors=10)
    }
    
    embeddings = {}
    
    for name, method in methods.items():
        print(f"\nTesting {name}...")
        try:
            if name == 'VAE':
                embedding = method.fit_transform(X, epochs=50, batch_size=64)
            else:
                embedding = method.fit_transform(X)
            
            embeddings[name] = embedding
            
            # Evaluate quality
            quality = evaluate_manifold_quality(X, embedding)
            print(f"  Trustworthiness: {quality['trustworthiness']:.3f}")
            print(f"  Continuity: {quality['continuity']:.3f}")
            print(f"  Stress: {quality['stress']:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test ensemble
    print("\nTesting Manifold Learning Ensemble...")
    try:
        ensemble = ManifoldLearningEnsemble(n_components=2, combine_method='average')
        combined_embedding = ensemble.fit_transform(X)
        
        quality = evaluate_manifold_quality(X, combined_embedding)
        print(f"Ensemble Quality:")
        print(f"  Trustworthiness: {quality['trustworthiness']:.3f}")
        print(f"  Continuity: {quality['continuity']:.3f}")
        print(f"  Stress: {quality['stress']:.3f}")
        
        # Plot results (if matplotlib available)
        try:
            ensemble.plot_embeddings(X, labels=color)
        except:
            print("Plotting not available")
    
    except Exception as e:
        print(f"Ensemble error: {e}")
    
    # Test with financial-like data
    print("\nTesting with Financial-like Data...")
    
    # Generate synthetic financial time series
    n_assets = 50
    n_periods = 252  # Trading days in a year
    
    # Random walk with correlations
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=0.02 * np.eye(n_assets) + 0.01 * np.ones((n_assets, n_assets)),
        size=n_periods
    )
    
    # Cumulative returns (price-like)
    prices = np.cumprod(1 + returns, axis=0)
    
    # Use rolling windows as features
    window_size = 20
    features = []
    
    for i in range(window_size, n_periods):
        window_data = prices[i-window_size:i].flatten()
        features.append(window_data)
    
    financial_data = np.array(features)
    print(f"Financial data shape: {financial_data.shape}")
    
    # Apply manifold learning
    try:
        vae = VariationalAutoencoder(
            input_dim=financial_data.shape[1],
            latent_dim=3,
            hidden_dims=[128, 64, 32]
        )
        
        financial_embedding = vae.fit_transform(financial_data, epochs=30)
        print(f"Financial embedding shape: {financial_embedding.shape}")
        
        # Evaluate
        quality = evaluate_manifold_quality(financial_data, financial_embedding)
        print(f"Financial Data Quality:")
        print(f"  Trustworthiness: {quality['trustworthiness']:.3f}")
        print(f"  Continuity: {quality['continuity']:.3f}")
        print(f"  Stress: {quality['stress']:.3f}")
        
    except Exception as e:
        print(f"Financial data test error: {e}")
    
    print("\nDone!")