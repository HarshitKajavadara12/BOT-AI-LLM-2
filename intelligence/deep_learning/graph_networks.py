"""
Graph Neural Networks for Financial Market Analysis
Advanced GNN architectures for modeling market relationships and dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import networkx as nx


class MarketGraphConv(MessagePassing):
    """
    Custom Graph Convolution for Financial Markets
    Incorporates market-specific edge features and temporal dynamics
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        aggr: str = 'add',
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Node transformation
        self.lin_node = nn.Linear(in_channels, out_channels, bias=bias)
        
        # Edge transformation
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)
        else:
            self.lin_edge = None
        
        # Attention mechanism for edge weighting
        self.attention = nn.Sequential(
            nn.Linear(2 * in_channels + (edge_dim or 0), out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_node.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        for layer in self.attention:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Node transformation
        out = self.lin_node(out)
        out = self.dropout(out)
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Compute attention weights
        if edge_attr is not None:
            attention_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            attention_input = torch.cat([x_i, x_j], dim=-1)
        
        alpha = self.attention(attention_input)
        
        # Apply edge transformation if available
        if self.lin_edge is not None and edge_attr is not None:
            edge_features = self.lin_edge(edge_attr)
            message = alpha * (x_j + edge_features)
        else:
            message = alpha * x_j
        
        return message


class CorrelationGraphNet(nn.Module):
    """
    Graph Neural Network based on asset correlation structure
    """
    
    def __init__(
        self,
        num_assets: int,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        correlation_threshold: float = 0.3
    ):
        super().__init__()
        self.num_assets = num_assets
        self.feature_dim = feature_dim
        self.correlation_threshold = correlation_threshold
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(MarketGraphConv(feature_dim, hidden_dim, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(MarketGraphConv(hidden_dim, hidden_dim, dropout=dropout))
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def build_correlation_graph(
        self,
        returns: torch.Tensor,
        lookback_window: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build correlation-based graph from return data
        
        Args:
            returns: Asset returns [batch_size, seq_len, num_assets]
            lookback_window: Window for correlation calculation
        
        Returns:
            edge_index: Graph edges [2, num_edges]
            edge_attr: Edge attributes (correlations) [num_edges, 1]
        """
        # Use most recent window for correlation
        recent_returns = returns[:, -lookback_window:, :]  # [batch, window, assets]
        
        # Compute correlation matrix
        corr_matrix = torch.corrcoef(recent_returns.mean(0).T)  # [assets, assets]
        
        # Threshold correlations and create edges
        mask = (torch.abs(corr_matrix) > self.correlation_threshold) & \
               (torch.eye(self.num_assets, device=corr_matrix.device) == 0)
        
        edge_indices = torch.nonzero(mask, as_tuple=False).T  # [2, num_edges]
        edge_weights = corr_matrix[mask].unsqueeze(-1)  # [num_edges, 1]
        
        return edge_indices, edge_weights
    
    def forward(
        self,
        x: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through correlation graph network
        
        Args:
            x: Node features [num_assets, feature_dim]
            returns: Historical returns for graph construction [batch, seq_len, num_assets]
            edge_index: Pre-computed edge indices [2, num_edges]
            edge_attr: Pre-computed edge attributes [num_edges, edge_dim]
        
        Returns:
            Asset predictions [num_assets, 1]
        """
        
        # Build graph if not provided
        if edge_index is None and returns is not None:
            edge_index, edge_attr = self.build_correlation_graph(returns)
        elif edge_index is None:
            # Fully connected graph as fallback
            edge_index = torch.combinations(
                torch.arange(self.num_assets, device=x.device), 2
            ).T
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_attr = None
        
        # Apply graph convolutions
        h = x
        for conv in self.convs:
            h_new = conv(h, edge_index, edge_attr)
            h = F.relu(h_new) + h if h.shape == h_new.shape else F.relu(h_new)
        
        h = self.layer_norm(h)
        
        # Output predictions
        output = self.output_layer(h)
        return output


class TemporalGraphNet(nn.Module):
    """
    Temporal Graph Neural Network for time-evolving market relationships
    """
    
    def __init__(
        self,
        num_assets: int,
        feature_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        num_temporal_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_assets = num_assets
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GATConv(feature_dim, hidden_dim, dropout=dropout))
        
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, dropout=dropout))
        
        # Temporal modeling (LSTM on graph embeddings)
        self.temporal_layers = nn.LSTM(
            hidden_dim, hidden_dim, num_temporal_layers,
            batch_first=True, dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index_seq: torch.Tensor,
        edge_attr_seq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through temporal graph network
        
        Args:
            x_seq: Node features over time [seq_len, num_assets, feature_dim]
            edge_index_seq: Edge indices over time [seq_len, 2, num_edges]
            edge_attr_seq: Edge attributes over time [seq_len, num_edges, edge_dim]
        
        Returns:
            Temporal predictions [seq_len, num_assets, 1]
        """
        seq_len, num_assets, _ = x_seq.shape
        
        # Process each time step through GNN
        graph_embeddings = []
        
        for t in range(seq_len):
            x_t = x_seq[t]  # [num_assets, feature_dim]
            edge_index_t = edge_index_seq[t]  # [2, num_edges]
            edge_attr_t = edge_attr_seq[t] if edge_attr_seq is not None else None
            
            # Apply GNN layers
            h = x_t
            for gnn_layer in self.gnn_layers:
                h = F.relu(gnn_layer(h, edge_index_t))
            
            graph_embeddings.append(h)
        
        # Stack temporal embeddings
        graph_embeddings = torch.stack(graph_embeddings, dim=0)  # [seq_len, num_assets, hidden_dim]
        
        # Process each asset's temporal sequence
        asset_predictions = []
        
        for asset_idx in range(num_assets):
            asset_sequence = graph_embeddings[:, asset_idx, :].unsqueeze(0)  # [1, seq_len, hidden_dim]
            
            # LSTM for temporal modeling
            lstm_out, _ = self.temporal_layers(asset_sequence)
            
            # Output prediction
            prediction = self.output_layer(lstm_out.squeeze(0))  # [seq_len, 1]
            asset_predictions.append(prediction)
        
        # Stack predictions
        predictions = torch.stack(asset_predictions, dim=1)  # [seq_len, num_assets, 1]
        
        return predictions


class HierarchicalGraphNet(nn.Module):
    """
    Hierarchical Graph Neural Network for multi-scale market analysis
    """
    
    def __init__(
        self,
        num_assets: int,
        feature_dim: int,
        hidden_dim: int = 128,
        num_levels: int = 3,
        pool_ratios: List[float] = [0.8, 0.6, 0.4],
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_assets = num_assets
        self.num_levels = num_levels
        self.pool_ratios = pool_ratios
        
        # Graph convolution layers for each level
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(feature_dim, hidden_dim))
        
        for level in range(1, num_levels):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Pooling layers
        self.pool_layers = nn.ModuleList()
        for level in range(num_levels - 1):
            self.pool_layers.append(
                nn.Linear(hidden_dim, int(hidden_dim * pool_ratios[level]))
            )
        
        # Readout layers
        self.global_pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Hierarchical graph processing
        level_embeddings = []
        h = x
        current_edge_index = edge_index
        
        for level in range(self.num_levels):
            # Graph convolution
            h = F.relu(self.conv_layers[level](h, current_edge_index))
            
            # Global pooling for this level
            level_embedding = self.global_pool(h, batch)
            level_embeddings.append(level_embedding)
            
            # Coarsening for next level (simple node sampling)
            if level < self.num_levels - 1:
                num_nodes_next = int(h.size(0) * self.pool_ratios[level])
                
                # Sample nodes (could be improved with learned pooling)
                sampled_nodes = torch.randperm(h.size(0), device=h.device)[:num_nodes_next]
                h = h[sampled_nodes]
                
                # Update edge index (simplified - real implementation would need proper coarsening)
                mask = torch.isin(current_edge_index[0], sampled_nodes) & \
                       torch.isin(current_edge_index[1], sampled_nodes)
                current_edge_index = current_edge_index[:, mask]
                
                # Remap node indices
                node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(sampled_nodes)}
                for i in range(current_edge_index.size(1)):
                    current_edge_index[0, i] = node_mapping.get(current_edge_index[0, i].item(), 0)
                    current_edge_index[1, i] = node_mapping.get(current_edge_index[1, i].item(), 0)
        
        # Concatenate embeddings from all levels
        combined_embedding = torch.cat(level_embeddings, dim=-1)
        
        # Final classification
        output = self.classifier(combined_embedding)
        return output


class GraphTransformer(nn.Module):
    """
    Graph Transformer for financial networks
    Combines graph structure with transformer attention
    """
    
    def __init__(
        self,
        num_assets: int,
        feature_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_layers.append(
                TransformerConv(
                    hidden_dim, hidden_dim, heads=num_heads,
                    dropout=dropout, concat=False
                )
            )
        
        # Position embedding for assets
        self.pos_embedding = nn.Parameter(torch.randn(num_assets, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # Input projection and position embedding
        h = self.input_proj(x) + self.pos_embedding
        h = self.dropout(h)
        
        # Apply transformer layers
        attention_weights = []
        for transformer_layer in self.transformer_layers:
            h_new = transformer_layer(h, edge_index, return_attention_weights=return_attention)
            
            if return_attention:
                h_new, attn = h_new
                attention_weights.append(attn)
            
            h = self.layer_norm(h_new + h)
        
        # Output prediction
        output = self.output_layer(h)
        
        if return_attention:
            return output, attention_weights
        return output


class DynamicGraphNet(nn.Module):
    """
    Dynamic Graph Neural Network that learns time-varying graph structure
    """
    
    def __init__(
        self,
        num_assets: int,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        edge_threshold: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_assets = num_assets
        self.edge_threshold = edge_threshold
        
        # Edge predictor network
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(SAGEConv(feature_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def predict_edges(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edges dynamically based on node features
        
        Args:
            x: Node features [num_assets, feature_dim]
        
        Returns:
            edge_index: Predicted edges [2, num_edges]
            edge_weights: Edge weights [num_edges]
        """
        num_assets = x.size(0)
        
        # Create all possible edge pairs
        all_pairs = torch.combinations(torch.arange(num_assets, device=x.device), 2)
        
        # Predict edge probabilities
        edge_features = torch.cat([
            x[all_pairs[:, 0]],  # Source node features
            x[all_pairs[:, 1]]   # Target node features
        ], dim=-1)
        
        edge_probs = self.edge_predictor(edge_features).squeeze(-1)
        
        # Threshold edges
        edge_mask = edge_probs > self.edge_threshold
        selected_pairs = all_pairs[edge_mask]
        selected_weights = edge_probs[edge_mask]
        
        # Create bidirectional edges
        edge_index = torch.cat([
            selected_pairs.T,
            selected_pairs.T.flip(0)
        ], dim=1)
        
        edge_weights = torch.cat([selected_weights, selected_weights])
        
        return edge_index, edge_weights
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamic graph structure
        
        Args:
            x: Node features [num_assets, feature_dim]
        
        Returns:
            predictions: Asset predictions [num_assets, 1]
            edge_index: Learned graph structure [2, num_edges]
        """
        
        # Predict dynamic graph structure
        edge_index, edge_weights = self.predict_edges(x)
        
        # Apply GNN layers
        h = x
        for gnn_layer in self.gnn_layers:
            h = F.relu(gnn_layer(h, edge_index))
            h = self.dropout(h)
        
        # Output predictions
        predictions = self.output_layer(h)
        
        return predictions, edge_index


def create_graph_model(
    model_type: str,
    num_assets: int,
    feature_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create graph neural network models
    
    Args:
        model_type: Type of model ('correlation', 'temporal', 'hierarchical', 'transformer', 'dynamic')
        num_assets: Number of assets/nodes
        feature_dim: Feature dimension
        **kwargs: Additional model-specific parameters
    
    Returns:
        Configured graph neural network model
    """
    
    if model_type == 'correlation':
        return CorrelationGraphNet(num_assets, feature_dim, **kwargs)
    elif model_type == 'temporal':
        return TemporalGraphNet(num_assets, feature_dim, **kwargs)
    elif model_type == 'hierarchical':
        return HierarchicalGraphNet(num_assets, feature_dim, **kwargs)
    elif model_type == 'transformer':
        return GraphTransformer(num_assets, feature_dim, **kwargs)
    elif model_type == 'dynamic':
        return DynamicGraphNet(num_assets, feature_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    num_assets, feature_dim = 50, 32
    x = torch.randn(num_assets, feature_dim)
    
    # Create simple edge index (random graph)
    num_edges = 100
    edge_index = torch.randint(0, num_assets, (2, num_edges))
    
    # Test different graph models
    models = {
        'correlation': CorrelationGraphNet(num_assets, feature_dim),
        'transformer': GraphTransformer(num_assets, feature_dim),
        'dynamic': DynamicGraphNet(num_assets, feature_dim)
    }
    
    for name, model in models.items():
        if name == 'correlation':
            # Need returns data for correlation model
            returns = torch.randn(1, 100, num_assets)
            output = model(x, returns=returns)
        elif name == 'dynamic':
            output, learned_edges = model(x)
            print(f"{name}: Input {x.shape} -> Output {output.shape}, Edges {learned_edges.shape}")
            continue
        else:
            output = model(x, edge_index)
        
        print(f"{name}: Input {x.shape} -> Output {output.shape}")
