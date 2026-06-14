"""
Advanced Attention Mechanisms for Financial Deep Learning
Specialized attention architectures for market data analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
from torch.nn.parameter import Parameter


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with financial-specific modifications
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        # Linear transformations for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class CrossAttention(nn.Module):
    """
    Cross-attention between different market data modalities
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(
        self,
        price_features: torch.Tensor,
        volume_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Cross-attention: price queries attend to volume keys/values
        output, _ = self.attention(
            query=price_features,
            key=volume_features,
            value=volume_features,
            mask=mask
        )
        
        return output


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for sequence modeling
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False
    ):
        super().__init__()
        self.causal = causal
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        mask = None
        if self.causal:
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            mask = mask.to(x.device)
        
        return self.attention(x, x, x, mask, return_attention)


class LocalAttention(nn.Module):
    """
    Local attention mechanism for high-frequency data
    Focuses on local temporal neighborhoods
    """
    
    def __init__(
        self,
        d_model: int,
        window_size: int = 10,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Apply local attention
        output = self._local_attention(Q, K, V)
        
        # Reshape and project
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + x)
    
    def _local_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, d_k = Q.shape
        output = torch.zeros_like(Q)
        
        for i in range(seq_len):
            # Define local window
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            # Local attention computation
            q_i = Q[:, i:i+1, :, :]  # [batch, 1, heads, d_k]
            k_local = K[:, start:end, :, :]  # [batch, window, heads, d_k]
            v_local = V[:, start:end, :, :]  # [batch, window, heads, d_k]
            
            # Compute attention scores
            scores = torch.matmul(q_i, k_local.transpose(-2, -1)) / math.sqrt(d_k)
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            
            # Apply attention
            output[:, i, :, :] = torch.matmul(weights, v_local).squeeze(1)
        
        return output


class SparseAttention(nn.Module):
    """
    Sparse attention mechanism for long sequences
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        block_size: int = 32,
        stride: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size
        self.stride = stride
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Create attention mask for sparse pattern
        mask = self._create_sparse_mask(seq_len, x.device)
        
        # Standard multi-head attention with sparse mask
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention with sparse mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(output)
        return self.layer_norm(output + x)
    
    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention mask"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        # Local attention blocks
        for i in range(0, seq_len, self.stride):
            start = i
            end = min(i + self.block_size, seq_len)
            mask[start:end, start:end] = 1
        
        # Global attention (every k-th position)
        global_positions = torch.arange(0, seq_len, step=self.stride, device=device)
        mask[global_positions, :] = 1
        mask[:, global_positions] = 1
        
        return mask.unsqueeze(0).unsqueeze(0)


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention that adjusts based on market conditions
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_conditions: int = 3  # e.g., low/medium/high volatility
    ):
        super().__init__()
        self.num_conditions = num_conditions
        self.d_model = d_model
        
        # Condition classifier
        self.condition_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_conditions)
        )
        
        # Separate attention for each condition
        self.attentions = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout)
            for _ in range(num_conditions)
        ])
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model + num_conditions, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Classify market condition based on recent context
        condition_input = x.mean(dim=1)  # Global average pooling
        condition_logits = self.condition_classifier(condition_input)
        condition_probs = F.softmax(condition_logits, dim=-1)
        
        # Apply attention for each condition
        attention_outputs = []
        for i, attention in enumerate(self.attentions):
            output, _ = attention(x, x, x)
            attention_outputs.append(output)
        
        # Weighted combination based on condition probabilities
        weighted_output = torch.zeros_like(x)
        for i, output in enumerate(attention_outputs):
            weight = condition_probs[:, i:i+1].unsqueeze(-1)
            weighted_output += weight * output
        
        # Adaptive gating
        gate_input = torch.cat([
            x.mean(dim=1, keepdim=True).expand(-1, seq_len, -1),
            condition_probs.unsqueeze(1).expand(-1, seq_len, -1)
        ], dim=-1)
        
        gate_values = self.gate(gate_input)
        
        # Apply gating
        output = gate_values * weighted_output + (1 - gate_values) * x
        
        return output


class TemporalAttention(nn.Module):
    """
    Temporal attention that emphasizes recent vs historical data
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        decay_factor: float = 0.95
    ):
        super().__init__()
        self.decay_factor = decay_factor
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Create temporal decay mask
        temporal_weights = torch.tensor([
            self.decay_factor ** (seq_len - 1 - i) for i in range(seq_len)
        ], device=x.device).unsqueeze(0).unsqueeze(0)
        
        # Apply attention with temporal weighting
        output, attention_weights = self.attention(
            x, x, x, return_attention=True
        )
        
        # Apply temporal decay to attention weights
        if attention_weights is not None:
            attention_weights = attention_weights * temporal_weights.unsqueeze(-1)
            
        return output


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention for multi-scale temporal patterns
    """
    
    def __init__(
        self,
        d_model: int,
        scales: list = [1, 5, 10, 20],
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.scales = scales
        
        # Attention layers for different scales
        self.scale_attentions = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout)
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_outputs = []
        
        for i, (scale, attention) in enumerate(zip(self.scales, self.scale_attentions)):
            # Downsample for this scale
            if scale > 1:
                downsampled = x[:, ::scale, :]
            else:
                downsampled = x
            
            # Apply attention
            scale_output, _ = attention(downsampled, downsampled, downsampled)
            
            # Upsample back if needed
            if scale > 1:
                # Interpolate to original length
                upsampled = F.interpolate(
                    scale_output.transpose(1, 2),
                    size=x.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                scale_outputs.append(upsampled)
            else:
                scale_outputs.append(scale_output)
        
        # Concatenate and fuse
        concatenated = torch.cat(scale_outputs, dim=-1)
        fused = self.scale_fusion(concatenated)
        
        return fused + x  # Residual connection


def create_attention_model(
    attention_type: str,
    d_model: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create attention models
    
    Args:
        attention_type: Type of attention ('multi_head', 'self', 'cross', 'local', 'sparse', 'adaptive', 'temporal', 'hierarchical')
        d_model: Model dimension
        **kwargs: Additional parameters
    
    Returns:
        Configured attention model
    """
    
    if attention_type == 'multi_head':
        return MultiHeadAttention(d_model, **kwargs)
    elif attention_type == 'self':
        return SelfAttention(d_model, **kwargs)
    elif attention_type == 'cross':
        return CrossAttention(d_model, **kwargs)
    elif attention_type == 'local':
        return LocalAttention(d_model, **kwargs)
    elif attention_type == 'sparse':
        return SparseAttention(d_model, **kwargs)
    elif attention_type == 'adaptive':
        return AdaptiveAttention(d_model, **kwargs)
    elif attention_type == 'temporal':
        return TemporalAttention(d_model, **kwargs)
    elif attention_type == 'hierarchical':
        return HierarchicalAttention(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


if __name__ == "__main__":
    # Example usage
    batch_size, seq_len, d_model = 32, 100, 256
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test different attention mechanisms
    attention_models = {
        'multi_head': MultiHeadAttention(d_model),
        'self': SelfAttention(d_model),
        'local': LocalAttention(d_model, window_size=20),
        'adaptive': AdaptiveAttention(d_model),
        'hierarchical': HierarchicalAttention(d_model)
    }
    
    for name, model in attention_models.items():
        if name == 'multi_head':
            output, _ = model(x, x, x)
        else:
            output = model(x)
        print(f"{name}: Input {x.shape} -> Output {output.shape}")
