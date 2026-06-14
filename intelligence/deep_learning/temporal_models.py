"""
Temporal Models for Financial Time Series Analysis
Advanced neural architectures for modeling temporal dependencies in market data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class LSTMAttentionModel(nn.Module):
    """
    LSTM with attention mechanism for financial time series prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.1,
        attention_dim: int = 64
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout
        )
        
        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)
        
        # Use last time step
        final_hidden = attn_out[:, -1, :]
        
        # Output prediction
        output = self.fc_out(final_hidden)
        return output


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for high-frequency trading signals
    """
    
    def __init__(
        self,
        input_channels: int,
        num_channels: list,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size, 
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalBlock(nn.Module):
    """
    Building block for Temporal Convolutional Network
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove extra padding from causal convolution"""
    
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TransformerPredictor(nn.Module):
    """
    Transformer-based model for financial sequence prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_length: int = 1000,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='gelu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input projection and positional encoding
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer encoding
        output = self.transformer_encoder(src.transpose(0, 1), src_mask)
        
        # Use last time step for prediction
        final_output = output[-1, :, :]  # [batch_size, d_model]
        
        # Output projection
        prediction = self.output_projection(final_output)
        return prediction


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class WaveNet(nn.Module):
    """
    WaveNet architecture adapted for financial time series
    """
    
    def __init__(
        self,
        input_channels: int,
        residual_channels: int = 32,
        skip_channels: int = 32,
        end_channels: int = 256,
        num_blocks: int = 4,
        num_layers: int = 10,
        output_length: int = 1,
        kernel_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        
        # Start convolution
        self.start_conv = nn.Conv1d(
            input_channels, residual_channels, kernel_size=1
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for block in range(num_blocks):
            for layer in range(num_layers):
                dilation = 2 ** layer
                self.residual_blocks.append(
                    ResidualBlock(
                        residual_channels, skip_channels,
                        kernel_size, dilation, dropout
                    )
                )
        
        # End convolutions
        self.end_conv_1 = nn.Conv1d(skip_channels, end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(end_channels, output_length, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start_conv(x)
        skip_connections = []
        
        for residual_block in self.residual_blocks:
            x, skip = residual_block(x)
            skip_connections.append(skip)
            
        # Sum skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # End convolutions
        x = F.relu(skip_sum)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        return x[:, :, -1]  # Return last time step


class ResidualBlock(nn.Module):
    """Residual block for WaveNet"""
    
    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels,
            kernel_size, dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        
        self.conv_1x1 = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dilated convolution
        conv_out = self.dilated_conv(x)
        
        # Gated activation
        filter_out, gate_out = conv_out.chunk(2, dim=1)
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        gated = self.dropout(gated)
        
        # Skip connection
        skip = self.skip_conv(gated)
        
        # Residual connection
        residual = self.conv_1x1(gated) + x
        
        return residual, skip


class MultiScaleTemporalModel(nn.Module):
    """
    Multi-scale temporal model for capturing different time horizons
    """
    
    def __init__(
        self,
        input_dim: int,
        scales: list = [1, 5, 10, 20, 50],
        hidden_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.scales = scales
        self.scale_models = nn.ModuleList()
        
        # Create model for each scale
        for scale in scales:
            model = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=scale, padding=scale//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.scale_models.append(model)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        
        scale_outputs = []
        for model in self.scale_models:
            scale_out = model(x)
            # Global average pooling
            scale_out = scale_out.mean(dim=2)
            scale_outputs.append(scale_out)
        
        # Concatenate all scales
        fused = torch.cat(scale_outputs, dim=1)
        
        # Final prediction
        output = self.fusion(fused)
        return output


def create_temporal_model(
    model_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create temporal models
    
    Args:
        model_type: Type of model ('lstm_attention', 'tcn', 'transformer', 'wavenet', 'multiscale')
        input_dim: Input feature dimension
        **kwargs: Additional model-specific parameters
    
    Returns:
        Configured temporal model
    """
    
    if model_type == 'lstm_attention':
        return LSTMAttentionModel(input_dim, **kwargs)
    elif model_type == 'tcn':
        return TemporalConvNet(input_dim, **kwargs)
    elif model_type == 'transformer':
        return TransformerPredictor(input_dim, **kwargs)
    elif model_type == 'wavenet':
        return WaveNet(input_dim, **kwargs)
    elif model_type == 'multiscale':
        return MultiScaleTemporalModel(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    batch_size, seq_len, input_dim = 32, 100, 50
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test different models
    models = {
        'lstm_attention': LSTMAttentionModel(input_dim, 64, 2, 1),
        'transformer': TransformerPredictor(input_dim, d_model=128, num_layers=3),
        'multiscale': MultiScaleTemporalModel(input_dim)
    }
    
    for name, model in models.items():
        output = model(x)
        print(f"{name}: Input {x.shape} -> Output {output.shape}")
