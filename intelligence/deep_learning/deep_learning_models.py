"""
Deep Learning Models for QUANTUM-FORGE
Implements advanced neural networks for trading including LSTMs, Transformers, CNNs, and GANs.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time
from collections import deque
import pickle
import json
warnings.filterwarnings('ignore')

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class ModelType(Enum):
    """Types of deep learning models."""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RESNET = "resnet"
    AUTOENCODER = "autoencoder"
    GAN = "gan"
    VAE = "vae"
    ATTENTION = "attention"

class TaskType(Enum):
    """Types of prediction tasks."""
    PRICE_PREDICTION = "price_prediction"
    DIRECTION_PREDICTION = "direction_prediction"
    VOLATILITY_PREDICTION = "volatility_prediction"
    REGIME_DETECTION = "regime_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    FEATURE_EXTRACTION = "feature_extraction"
    SIGNAL_GENERATION = "signal_generation"

@dataclass
class ModelConfig:
    """Configuration for deep learning models."""
    model_type: ModelType
    task_type: TaskType
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    sequence_length: int
    epochs: int
    regularization: float

@dataclass
class TrainingResult:
    """Results from model training."""
    model_name: str
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    best_epoch: int
    training_time: float
    final_metrics: Dict

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data."""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, 
                 sequence_length: int, transform=None):
        """Initialize time series dataset."""
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.transform = transform
        
    def __len__(self):
        return len(self.data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        """Get sequence and target."""
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length - 1]
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2):
        """Initialize LSTM model."""
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """Forward pass."""
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden_dim)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, hidden_dim)
        
        # Use last time step
        last_output = attn_out[:, -1, :]
        
        # Layer normalization
        last_output = self.layer_norm(last_output)
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TransformerModel(nn.Module):
    """Transformer model for sequence prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int,
                 num_layers: int, output_dim: int, dropout: float = 0.1):
        """Initialize Transformer model."""
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """Forward pass."""
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer requires (seq_len, batch, features)
        x = x.transpose(0, 1)
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Use last time step
        last_output = transformer_out[-1]  # (batch, hidden_dim)
        
        # Layer normalization
        last_output = self.layer_norm(last_output)
        
        # Output layers
        out = self.dropout(last_output)
        out = F.gelu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding."""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class GRUModel(nn.Module):
    """GRU model for time series prediction — proper implementation (not CNN alias)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, dropout: float = 0.2):
        """Initialize GRU model."""
        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """Forward pass."""
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        gru_out, hn = self.gru(x, h0)

        # Use last time step
        last_output = gru_out[:, -1, :]
        last_output = self.layer_norm(last_output)

        out = self.dropout(last_output)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class CNNModel(nn.Module):
    """1D CNN model for time series."""
    
    def __init__(self, input_dim: int, num_filters: int, filter_sizes: List[int],
                 output_dim: int, dropout: float = 0.2):
        """Initialize CNN model."""
        super(CNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_filters = num_filters
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in filter_sizes
        ])
        
        # Pooling layers
        self.pools = nn.ModuleList([
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            for _ in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size
        total_filters = num_filters * len(filter_sizes)
        
        # Fully connected layers
        self.fc1 = nn.Linear(total_filters, total_filters // 2)
        self.fc2 = nn.Linear(total_filters // 2, output_dim)
        
    def forward(self, x):
        """Forward pass."""
        # Transpose for conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        conv_outputs = []
        
        for conv, bn, pool in zip(self.convs, self.batch_norms, self.pools):
            # Convolution
            conv_out = F.relu(bn(conv(x)))
            
            # Global max pooling
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        combined = torch.cat(conv_outputs, dim=1)
        
        # Fully connected layers
        out = self.dropout(combined)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class AutoEncoder(nn.Module):
    """Autoencoder for feature learning and anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_dims: List[int],
                 dropout: float = 0.2):
        """Initialize autoencoder."""
        super(AutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class GANGenerator(nn.Module):
    """Generator network for GAN."""
    
    def __init__(self, noise_dim: int, output_dim: int, hidden_dims: List[int]):
        """Initialize generator."""
        super(GANGenerator, self).__init__()
        
        layers = []
        prev_dim = noise_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, noise):
        """Generate samples."""
        return self.network(noise)

class GANDiscriminator(nn.Module):
    """Discriminator network for GAN."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """Initialize discriminator."""
        super(GANDiscriminator, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Discriminate between real and fake."""
        return self.network(x)

class DeepLearningEngine:
    """Main deep learning engine for trading models."""
    
    def __init__(self):
        """Initialize deep learning engine."""
        self.models = {}
        self.training_history = {}
        self.device = DEVICE
        
    def create_model(self, config: ModelConfig, model_name: str) -> nn.Module:
        """Create model based on configuration."""
        
        if config.model_type == ModelType.LSTM:
            model = LSTMModel(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                output_dim=config.output_dim,
                dropout=config.dropout
            )
            
        elif config.model_type == ModelType.TRANSFORMER:
            model = TransformerModel(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_heads=8,
                num_layers=config.num_layers,
                output_dim=config.output_dim,
                dropout=config.dropout
            )
            
        elif config.model_type == ModelType.CNN:
            model = CNNModel(
                input_dim=config.input_dim,
                num_filters=config.hidden_dim,
                filter_sizes=[3, 5, 7],
                output_dim=config.output_dim,
                dropout=config.dropout
            )
            
        elif config.model_type == ModelType.AUTOENCODER:
            hidden_dims = [config.hidden_dim, config.hidden_dim // 2]
            model = AutoEncoder(
                input_dim=config.input_dim,
                encoding_dim=config.output_dim,
                hidden_dims=hidden_dims,
                dropout=config.dropout
            )
            
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        model = model.to(self.device)
        self.models[model_name] = {
            'model': model,
            'config': config,
            'optimizer': None,
            'scheduler': None
        }
        
        return model
    
    def train_model(self, model_name: str, train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None,
                   epochs: Optional[int] = None) -> TrainingResult:
        """Train deep learning model."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.models[model_name]
        model = model_info['model']
        config = model_info['config']
        
        if epochs is None:
            epochs = config.epochs
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                             weight_decay=config.regularization)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        model_info['optimizer'] = optimizer
        model_info['scheduler'] = scheduler
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        best_epoch = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if config.model_type == ModelType.AUTOENCODER:
                    output, _ = model(data.view(data.size(0), -1))
                    loss = F.mse_loss(output, data.view(data.size(0), -1))
                    
                    # For autoencoder, accuracy is reconstruction quality
                    with torch.no_grad():
                        mse = F.mse_loss(output, data.view(data.size(0), -1), reduction='none')
                        reconstruction_error = mse.mean(dim=1)
                        threshold = reconstruction_error.median()
                        predictions = (reconstruction_error < threshold).float()
                        train_correct += predictions.sum().item()
                        
                else:
                    output = model(data)
                    
                    if config.task_type == TaskType.DIRECTION_PREDICTION:
                        # Classification task
                        target = target.long()
                        loss = F.cross_entropy(output, target)
                        
                        # Calculate accuracy
                        with torch.no_grad():
                            _, predicted = torch.max(output.data, 1)
                            train_correct += (predicted == target).sum().item()
                    else:
                        # Regression task
                        loss = F.mse_loss(output, target)
                        
                        # For regression, use R² as accuracy metric
                        with torch.no_grad():
                            ss_res = ((target - output) ** 2).sum()
                            ss_tot = ((target - target.mean()) ** 2).sum()
                            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                            train_correct += r_squared.item() * target.size(0)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                train_total += target.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation phase
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        if config.model_type == ModelType.AUTOENCODER:
                            output, _ = model(data.view(data.size(0), -1))
                            loss = F.mse_loss(output, data.view(data.size(0), -1))
                            
                            mse = F.mse_loss(output, data.view(data.size(0), -1), reduction='none')
                            reconstruction_error = mse.mean(dim=1)
                            threshold = reconstruction_error.median()
                            predictions = (reconstruction_error < threshold).float()
                            val_correct += predictions.sum().item()
                            
                        else:
                            output = model(data)
                            
                            if config.task_type == TaskType.DIRECTION_PREDICTION:
                                target = target.long()
                                loss = F.cross_entropy(output, target)
                                
                                _, predicted = torch.max(output.data, 1)
                                val_correct += (predicted == target).sum().item()
                            else:
                                loss = F.mse_loss(output, target)
                                
                                ss_res = ((target - output) ** 2).sum()
                                ss_tot = ((target - target.mean()) ** 2).sum()
                                r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                                val_correct += r_squared.item() * target.size(0)
                        
                        val_loss += loss.item()
                        val_total += target.size(0)
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                val_losses.append(avg_val_loss)
                val_accuracies.append(val_accuracy)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'loss': avg_val_loss
                    }, f'best_{model_name}.pth')
            
            # Print progress
            if epoch % 10 == 0:
                if val_loader is not None:
                    print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                          f'Val Acc: {val_accuracy:.4f}')
                else:
                    print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                          f'Train Acc: {train_accuracy:.4f}')
        
        training_time = time.time() - start_time
        
        # Create training result
        final_metrics = {
            'final_train_loss': train_losses[-1],
            'final_train_accuracy': train_accuracies[-1],
            'best_val_loss': best_val_loss if val_loader else None,
            'best_val_accuracy': val_accuracies[best_epoch] if val_loader else None,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        result = TrainingResult(
            model_name=model_name,
            train_loss=train_losses,
            val_loss=val_losses,
            train_accuracy=train_accuracies,
            val_accuracy=val_accuracies,
            best_epoch=best_epoch,
            training_time=training_time,
            final_metrics=final_metrics
        )
        
        self.training_history[model_name] = result
        return result
    
    def predict(self, model_name: str, data: torch.Tensor) -> torch.Tensor:
        """Make predictions with trained model."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]['model']
        config = self.models[model_name]['config']
        
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            if config.model_type == ModelType.AUTOENCODER:
                output, encoding = model(data.view(data.size(0), -1))
                return output, encoding
            else:
                output = model(data)
                return output
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.models[model_name]
        
        torch.save({
            'model_state_dict': model_info['model'].state_dict(),
            'config': model_info['config'],
            'optimizer_state_dict': model_info['optimizer'].state_dict() if model_info['optimizer'] else None,
            'training_history': self.training_history.get(model_name, None)
        }, filepath)
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        config = checkpoint['config']
        
        # Create model
        model = self.create_model(config, model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer if available
        if checkpoint.get('optimizer_state_dict'):
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.models[model_name]['optimizer'] = optimizer
        
        # Load training history
        if checkpoint.get('training_history'):
            self.training_history[model_name] = checkpoint['training_history']
    
    def get_model_summary(self, model_name: str) -> Dict:
        """Get model summary and statistics."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]['model']
        config = self.models[model_name]['config']
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'model_name': model_name,
            'model_type': config.model_type.value,
            'task_type': config.task_type.value,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'config': config,
            'device': str(next(model.parameters()).device)
        }
        
        if model_name in self.training_history:
            history = self.training_history[model_name]
            summary.update({
                'training_completed': True,
                'best_epoch': history.best_epoch,
                'training_time': history.training_time,
                'final_metrics': history.final_metrics
            })
        else:
            summary['training_completed'] = False
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    print("Testing Deep Learning Models...")
    
    # Generate synthetic financial time series data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters
    n_samples = 10000
    sequence_length = 60
    n_features = 5
    
    # Generate realistic financial time series
    prices = np.zeros((n_samples, n_features))
    prices[0] = [100, 50, 25, 150, 75]  # Initial prices
    
    for i in range(1, n_samples):
        # Add random walk with drift and volatility
        returns = np.random.multivariate_normal(
            mean=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
            cov=np.diag([0.0001, 0.0002, 0.0003, 0.0001, 0.0002])  # Different volatilities
        )
        prices[i] = prices[i-1] * (1 + returns)
    
    # Create features (returns, volatility, etc.)
    returns = np.diff(prices, axis=0) / prices[:-1]
    volatility = np.array([np.std(returns[max(0, i-20):i+1], axis=0) for i in range(len(returns))])
    
    # Combine features
    features = np.concatenate([
        returns,
        volatility,
        np.roll(returns, 1, axis=0),  # Lagged returns
        np.roll(volatility, 1, axis=0)  # Lagged volatility
    ], axis=1)
    
    features = features[20:]  # Remove initial NaN values
    
    print(f"Generated dataset: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Create targets for different tasks
    # Price prediction target (next period return)
    price_targets = returns[21:, 0]  # Predict first asset return
    
    # Direction prediction target (binary: up/down)
    direction_targets = (price_targets > 0).astype(int)
    
    print(f"Price targets: {len(price_targets)} samples")
    print(f"Direction targets: {len(direction_targets)} samples, Up: {np.sum(direction_targets)}")
    
    # Create datasets
    price_dataset = TimeSeriesDataset(features, price_targets, sequence_length)
    direction_dataset = TimeSeriesDataset(features, direction_targets, sequence_length)
    
    # Split data
    train_size = int(0.8 * len(price_dataset))
    val_size = len(price_dataset) - train_size
    
    price_train, price_val = torch.utils.data.random_split(price_dataset, [train_size, val_size])
    direction_train, direction_val = torch.utils.data.random_split(direction_dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 32
    price_train_loader = DataLoader(price_train, batch_size=batch_size, shuffle=True)
    price_val_loader = DataLoader(price_val, batch_size=batch_size, shuffle=False)
    direction_train_loader = DataLoader(direction_train, batch_size=batch_size, shuffle=True)
    direction_val_loader = DataLoader(direction_val, batch_size=batch_size, shuffle=False)
    
    print(f"Data loaders created: {len(price_train_loader)} train batches, {len(price_val_loader)} val batches")
    
    # Initialize deep learning engine
    dl_engine = DeepLearningEngine()
    
    # Test 1: LSTM for price prediction
    print("\n=== Testing LSTM for Price Prediction ===")
    
    lstm_config = ModelConfig(
        model_type=ModelType.LSTM,
        task_type=TaskType.PRICE_PREDICTION,
        input_dim=features.shape[1],
        output_dim=1,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=batch_size,
        sequence_length=sequence_length,
        epochs=50,
        regularization=1e-5
    )
    
    lstm_model = dl_engine.create_model(lstm_config, "lstm_price_predictor")
    lstm_result = dl_engine.train_model("lstm_price_predictor", price_train_loader, price_val_loader, epochs=20)
    
    print(f"LSTM Training Results:")
    print(f"  Final train loss: {lstm_result.final_metrics['final_train_loss']:.6f}")
    print(f"  Final train R²: {lstm_result.final_metrics['final_train_accuracy']:.4f}")
    print(f"  Best val loss: {lstm_result.final_metrics['best_val_loss']:.6f}")
    print(f"  Best val R²: {lstm_result.final_metrics['best_val_accuracy']:.4f}")
    print(f"  Training time: {lstm_result.training_time:.2f}s")
    
    # Test 2: Transformer for direction prediction
    print("\n=== Testing Transformer for Direction Prediction ===")
    
    transformer_config = ModelConfig(
        model_type=ModelType.TRANSFORMER,
        task_type=TaskType.DIRECTION_PREDICTION,
        input_dim=features.shape[1],
        output_dim=2,  # Binary classification
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
        learning_rate=0.0005,
        batch_size=batch_size,
        sequence_length=sequence_length,
        epochs=30,
        regularization=1e-4
    )
    
    transformer_model = dl_engine.create_model(transformer_config, "transformer_direction_predictor")
    transformer_result = dl_engine.train_model("transformer_direction_predictor", direction_train_loader, direction_val_loader, epochs=15)
    
    print(f"Transformer Training Results:")
    print(f"  Final train loss: {transformer_result.final_metrics['final_train_loss']:.6f}")
    print(f"  Final train accuracy: {transformer_result.final_metrics['final_train_accuracy']:.4f}")
    print(f"  Best val loss: {transformer_result.final_metrics['best_val_loss']:.6f}")
    print(f"  Best val accuracy: {transformer_result.final_metrics['best_val_accuracy']:.4f}")
    print(f"  Training time: {transformer_result.training_time:.2f}s")
    
    # Test 3: CNN for feature extraction
    print("\n=== Testing CNN for Price Prediction ===")
    
    cnn_config = ModelConfig(
        model_type=ModelType.CNN,
        task_type=TaskType.PRICE_PREDICTION,
        input_dim=features.shape[1],
        output_dim=1,
        hidden_dim=32,  # Number of filters
        num_layers=3,
        dropout=0.3,
        learning_rate=0.002,
        batch_size=batch_size,
        sequence_length=sequence_length,
        epochs=40,
        regularization=1e-4
    )
    
    cnn_model = dl_engine.create_model(cnn_config, "cnn_price_predictor")
    cnn_result = dl_engine.train_model("cnn_price_predictor", price_train_loader, price_val_loader, epochs=15)
    
    print(f"CNN Training Results:")
    print(f"  Final train loss: {cnn_result.final_metrics['final_train_loss']:.6f}")
    print(f"  Final train R²: {cnn_result.final_metrics['final_train_accuracy']:.4f}")
    print(f"  Best val loss: {cnn_result.final_metrics['best_val_loss']:.6f}")
    print(f"  Best val R²: {cnn_result.final_metrics['best_val_accuracy']:.4f}")
    print(f"  Training time: {cnn_result.training_time:.2f}s")
    
    # Test 4: Autoencoder for anomaly detection
    print("\n=== Testing Autoencoder for Feature Learning ===")
    
    # Flatten features for autoencoder
    flattened_features = features.reshape(features.shape[0], -1)
    autoencoder_dataset = TensorDataset(
        torch.FloatTensor(flattened_features),
        torch.FloatTensor(flattened_features)  # Autoencoder targets are inputs
    )
    
    ae_train_size = int(0.8 * len(autoencoder_dataset))
    ae_val_size = len(autoencoder_dataset) - ae_train_size
    ae_train, ae_val = torch.utils.data.random_split(autoencoder_dataset, [ae_train_size, ae_val_size])
    
    ae_train_loader = DataLoader(ae_train, batch_size=batch_size, shuffle=True)
    ae_val_loader = DataLoader(ae_val, batch_size=batch_size, shuffle=False)
    
    ae_config = ModelConfig(
        model_type=ModelType.AUTOENCODER,
        task_type=TaskType.FEATURE_EXTRACTION,
        input_dim=flattened_features.shape[1],
        output_dim=32,  # Encoding dimension
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=batch_size,
        sequence_length=1,  # Not used for autoencoder
        epochs=50,
        regularization=1e-5
    )
    
    ae_model = dl_engine.create_model(ae_config, "feature_autoencoder")
    ae_result = dl_engine.train_model("feature_autoencoder", ae_train_loader, ae_val_loader, epochs=20)
    
    print(f"Autoencoder Training Results:")
    print(f"  Final train loss: {ae_result.final_metrics['final_train_loss']:.6f}")
    print(f"  Final train accuracy: {ae_result.final_metrics['final_train_accuracy']:.4f}")
    print(f"  Best val loss: {ae_result.final_metrics['best_val_loss']:.6f}")
    print(f"  Best val accuracy: {ae_result.final_metrics['best_val_accuracy']:.4f}")
    print(f"  Training time: {ae_result.training_time:.2f}s")
    
    # Test predictions
    print("\n=== Testing Model Predictions ===")
    
    # Sample test data
    test_data = torch.FloatTensor(features[:sequence_length]).unsqueeze(0)  # Batch size 1
    
    # LSTM prediction
    lstm_pred = dl_engine.predict("lstm_price_predictor", test_data)
    print(f"LSTM prediction: {lstm_pred.item():.6f}")
    
    # Transformer prediction
    transformer_pred = dl_engine.predict("transformer_direction_predictor", test_data)
    transformer_probs = F.softmax(transformer_pred, dim=1)
    print(f"Transformer prediction: {transformer_probs.numpy()[0]} (probabilities)")
    
    # CNN prediction
    cnn_pred = dl_engine.predict("cnn_price_predictor", test_data)
    print(f"CNN prediction: {cnn_pred.item():.6f}")
    
    # Autoencoder prediction
    ae_test_data = torch.FloatTensor(flattened_features[:1])
    ae_pred, ae_encoding = dl_engine.predict("feature_autoencoder", ae_test_data)
    reconstruction_error = F.mse_loss(ae_pred, ae_test_data).item()
    print(f"Autoencoder reconstruction error: {reconstruction_error:.6f}")
    print(f"Encoding dimension: {ae_encoding.shape[1]}")
    
    # Model summaries
    print("\n=== Model Summaries ===")
    
    for model_name in dl_engine.models.keys():
        summary = dl_engine.get_model_summary(model_name)
        print(f"\n{model_name}:")
        print(f"  Model type: {summary['model_type']}")
        print(f"  Task type: {summary['task_type']}")
        print(f"  Total parameters: {summary['total_parameters']:,}")
        print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
        print(f"  Model size: {summary['model_size_mb']:.2f} MB")
        print(f"  Device: {summary['device']}")
        
        if summary['training_completed']:
            print(f"  Best epoch: {summary['best_epoch']}")
            print(f"  Training time: {summary['training_time']:.2f}s")
    
    print("\nDeep learning models testing completed successfully!")