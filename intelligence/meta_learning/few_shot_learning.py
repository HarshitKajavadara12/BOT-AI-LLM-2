"""
Few-Shot Learning for Financial Models
Advanced meta-learning techniques for quick adaptation to new market conditions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from abc import ABC, abstractmethod
import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error


class FewShotLearner(ABC):
    """Abstract base class for few-shot learning methods"""
    
    @abstractmethod
    def meta_train(
        self,
        support_sets: List[Tuple[torch.Tensor, torch.Tensor]],
        query_sets: List[Tuple[torch.Tensor, torch.Tensor]],
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """
        Meta-training on multiple tasks
        
        Args:
            support_sets: List of (X_support, y_support) for each task
            query_sets: List of (X_query, y_query) for each task
            n_epochs: Number of meta-training epochs
        
        Returns:
            Training metrics
        """
        pass
    
    @abstractmethod
    def adapt(
        self,
        X_support: torch.Tensor,
        y_support: torch.Tensor,
        n_steps: int = 10
    ) -> None:
        """
        Adapt to new task with few examples
        
        Args:
            X_support: Support set features
            y_support: Support set labels
            n_steps: Number of adaptation steps
        """
        pass
    
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on new data
        
        Args:
            X: Input features
        
        Returns:
            Predictions
        """
        pass


class MAML(FewShotLearner):
    """
    Model-Agnostic Meta-Learning (MAML)
    Learns initialization parameters for fast adaptation
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
        first_order: bool = False
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.initial_params = None
        
    def _inner_loop(
        self,
        X_support: torch.Tensor,
        y_support: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inner loop adaptation for a single task
        
        Args:
            X_support: Support set features
            y_support: Support set labels  
            params: Current parameters (if None, use model parameters)
        
        Returns:
            Updated parameters
        """
        
        if params is None:
            params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Inner loop updates
        for step in range(self.inner_steps):
            # Forward pass with current parameters
            logits = self._forward_with_params(X_support, params)
            
            # Compute loss
            if y_support.dtype == torch.long:
                # Classification
                loss = F.cross_entropy(logits, y_support)
            else:
                # Regression
                loss = F.mse_loss(logits.squeeze(), y_support)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=not self.first_order,
                retain_graph=True,
                allow_unused=True
            )
            
            # Update parameters
            updated_params = {}
            for (name, param), grad in zip(params.items(), grads):
                if grad is not None:
                    updated_params[name] = param - self.inner_lr * grad
                else:
                    updated_params[name] = param
            
            params = updated_params
        
        return params
    
    def _forward_with_params(
        self,
        X: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with custom parameters"""
        
        # This is a simplified implementation
        # In practice, you'd need to implement functional versions of your layers
        x = X
        
        # Example for simple MLP
        if 'fc1.weight' in params and 'fc1.bias' in params:
            x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
            x = F.relu(x)
        
        if 'fc2.weight' in params and 'fc2.bias' in params:
            x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
            x = F.relu(x)
        
        if 'output.weight' in params and 'output.bias' in params:
            x = F.linear(x, params['output.weight'], params['output.bias'])
        
        return x
    
    def meta_train(
        self,
        support_sets: List[Tuple[torch.Tensor, torch.Tensor]],
        query_sets: List[Tuple[torch.Tensor, torch.Tensor]],
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """Meta-training with MAML"""
        
        self.model.train()
        meta_losses = []
        
        for epoch in range(n_epochs):
            meta_loss = 0.0
            
            # Sample batch of tasks
            for (X_support, y_support), (X_query, y_query) in zip(support_sets, query_sets):
                
                # Inner loop adaptation
                adapted_params = self._inner_loop(X_support, y_support)
                
                # Meta loss on query set
                query_logits = self._forward_with_params(X_query, adapted_params)
                
                if y_query.dtype == torch.long:
                    task_loss = F.cross_entropy(query_logits, y_query)
                else:
                    task_loss = F.mse_loss(query_logits.squeeze(), y_query)
                
                meta_loss += task_loss
            
            meta_loss /= len(support_sets)
            
            # Meta-update
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            meta_losses.append(meta_loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Meta Loss: {meta_loss.item():.4f}")
        
        return {'meta_losses': meta_losses}
    
    def adapt(
        self,
        X_support: torch.Tensor,
        y_support: torch.Tensor,
        n_steps: int = 10
    ) -> None:
        """Adapt to new task"""
        
        # Save initial parameters
        self.initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Adapt parameters
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            logits = self.model(X_support)
            
            if y_support.dtype == torch.long:
                loss = F.cross_entropy(logits, y_support)
            else:
                loss = F.mse_loss(logits.squeeze(), y_support)
            
            loss.backward()
            optimizer.step()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            return self.model(X)
    
    def reset_to_initial(self):
        """Reset model to initial meta-learned parameters"""
        if self.initial_params is not None:
            for name, param in self.model.named_parameters():
                param.data.copy_(self.initial_params[name])


class ProtoNet(FewShotLearner):
    """
    Prototypical Networks for few-shot learning
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        distance_metric: str = 'euclidean',
        learning_rate: float = 0.001
    ):
        self.encoder = encoder
        self.distance_metric = distance_metric
        self.learning_rate = learning_rate
        
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.prototypes = None
        self.classes = None
        
    def _compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute distance between embeddings"""
        
        if self.distance_metric == 'euclidean':
            return torch.cdist(x, y, p=2)
        elif self.distance_metric == 'cosine':
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            return 1 - torch.mm(x_norm, y_norm.t())
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _compute_prototypes(
        self,
        X_support: torch.Tensor,
        y_support: torch.Tensor
    ) -> torch.Tensor:
        """Compute class prototypes from support set"""
        
        # Encode support set
        support_embeddings = self.encoder(X_support)
        
        # Compute prototypes for each class
        classes = torch.unique(y_support)
        prototypes = []
        
        for cls in classes:
            class_mask = (y_support == cls)
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes), classes
    
    def meta_train(
        self,
        support_sets: List[Tuple[torch.Tensor, torch.Tensor]],
        query_sets: List[Tuple[torch.Tensor, torch.Tensor]],
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """Meta-training with Prototypical Networks"""
        
        self.encoder.train()
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for (X_support, y_support), (X_query, y_query) in zip(support_sets, query_sets):
                
                self.optimizer.zero_grad()
                
                # Compute prototypes
                prototypes, classes = self._compute_prototypes(X_support, y_support)
                
                # Encode query set
                query_embeddings = self.encoder(X_query)
                
                # Compute distances to prototypes
                distances = self._compute_distance(query_embeddings, prototypes)
                
                # Convert to logits (negative distances)
                logits = -distances
                
                # Create target labels (indices into classes)
                targets = torch.zeros_like(y_query, dtype=torch.long)
                for i, cls in enumerate(classes):
                    targets[y_query == cls] = i
                
                # Compute loss
                loss = F.cross_entropy(logits, targets)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(support_sets))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
        
        return {'losses': losses}
    
    def adapt(
        self,
        X_support: torch.Tensor,
        y_support: torch.Tensor,
        n_steps: int = 10
    ) -> None:
        """Adapt by computing prototypes"""
        
        self.encoder.eval()
        with torch.no_grad():
            self.prototypes, self.classes = self._compute_prototypes(X_support, y_support)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using prototypes"""
        
        if self.prototypes is None:
            raise ValueError("Must adapt to support set before prediction")
        
        self.encoder.eval()
        with torch.no_grad():
            query_embeddings = self.encoder(X)
            distances = self._compute_distance(query_embeddings, self.prototypes)
            
            # Return class with minimum distance
            min_distances, class_indices = torch.min(distances, dim=1)
            predictions = self.classes[class_indices]
            
            return predictions


class RelationNet(FewShotLearner):
    """
    Relation Networks for few-shot learning
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        relation_module: nn.Module,
        learning_rate: float = 0.001
    ):
        self.encoder = encoder
        self.relation_module = relation_module
        self.learning_rate = learning_rate
        
        # Combined optimizer for both modules
        params = list(encoder.parameters()) + list(relation_module.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        self.support_embeddings = None
        self.support_labels = None
    
    def _relation_score(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute relation scores between query and support embeddings"""
        
        n_query = query_embeddings.size(0)
        n_support = support_embeddings.size(0)
        
        # Expand embeddings for pairwise comparison
        query_expanded = query_embeddings.unsqueeze(1).expand(n_query, n_support, -1)
        support_expanded = support_embeddings.unsqueeze(0).expand(n_query, n_support, -1)
        
        # Concatenate embeddings
        relation_pairs = torch.cat([query_expanded, support_expanded], dim=2)
        
        # Compute relation scores
        relation_scores = self.relation_module(relation_pairs.view(-1, relation_pairs.size(2)))
        relation_scores = relation_scores.view(n_query, n_support)
        
        return relation_scores
    
    def meta_train(
        self,
        support_sets: List[Tuple[torch.Tensor, torch.Tensor]],
        query_sets: List[Tuple[torch.Tensor, torch.Tensor]],
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """Meta-training with Relation Networks"""
        
        self.encoder.train()
        self.relation_module.train()
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for (X_support, y_support), (X_query, y_query) in zip(support_sets, query_sets):
                
                self.optimizer.zero_grad()
                
                # Encode sets
                support_embeddings = self.encoder(X_support)
                query_embeddings = self.encoder(X_query)
                
                # Compute relation scores
                relation_scores = self._relation_score(query_embeddings, support_embeddings)
                
                # Create targets (1 if same class, 0 if different)
                targets = (y_query.unsqueeze(1) == y_support.unsqueeze(0)).float()
                
                # Binary classification loss for each query-support pair
                loss = F.binary_cross_entropy_with_logits(relation_scores, targets)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(support_sets))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
        
        return {'losses': losses}
    
    def adapt(
        self,
        X_support: torch.Tensor,
        y_support: torch.Tensor,
        n_steps: int = 10
    ) -> None:
        """Adapt by storing support set embeddings"""
        
        self.encoder.eval()
        with torch.no_grad():
            self.support_embeddings = self.encoder(X_support)
            self.support_labels = y_support
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using relation scores"""
        
        if self.support_embeddings is None:
            raise ValueError("Must adapt to support set before prediction")
        
        self.encoder.eval()
        self.relation_module.eval()
        
        with torch.no_grad():
            query_embeddings = self.encoder(X)
            relation_scores = self._relation_score(query_embeddings, self.support_embeddings)
            
            # Find support example with highest relation score for each query
            _, max_indices = torch.max(relation_scores, dim=1)
            predictions = self.support_labels[max_indices]
            
            return predictions


class FinancialFewShotLearner:
    """
    Financial-specific few-shot learning wrapper
    Handles financial time series and market regime adaptation
    """
    
    def __init__(
        self,
        base_learner: FewShotLearner,
        feature_extractor: Optional[nn.Module] = None,
        task_type: str = 'classification'  # 'classification' or 'regression'
    ):
        self.base_learner = base_learner
        self.feature_extractor = feature_extractor
        self.task_type = task_type
        
        self.scaler = None
        self.regime_history = []
        
    def _extract_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract financial features from raw data"""
        
        if self.feature_extractor is None:
            return X
        
        return self.feature_extractor(X)
    
    def _create_financial_tasks(
        self,
        data: Dict[str, pd.DataFrame],
        window_size: int = 20,
        n_tasks: int = 10
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Create few-shot learning tasks from financial data
        
        Args:
            data: Dictionary of DataFrames (e.g., different assets or time periods)
            window_size: Size of input window
            n_tasks: Number of tasks to create
        
        Returns:
            Support and query sets for each task
        """
        
        support_sets = []
        query_sets = []
        
        for i in range(n_tasks):
            # Randomly select asset/market
            asset_name = np.random.choice(list(data.keys()))
            asset_data = data[asset_name]
            
            # Create windows
            n_samples = len(asset_data) - window_size
            if n_samples < 20:  # Need minimum samples
                continue
            
            # Random split for support/query
            indices = np.random.permutation(n_samples)
            n_support = n_samples // 2
            
            support_indices = indices[:n_support]
            query_indices = indices[n_support:n_support*2]  # Equal size
            
            # Create features and targets
            X_support = []
            y_support = []
            X_query = []
            y_query = []
            
            for idx in support_indices:
                window = asset_data.iloc[idx:idx+window_size].values
                
                if self.task_type == 'classification':
                    # Predict direction (up/down)
                    target = 1 if asset_data.iloc[idx+window_size]['close'] > asset_data.iloc[idx+window_size-1]['close'] else 0
                else:
                    # Predict return
                    target = (asset_data.iloc[idx+window_size]['close'] / asset_data.iloc[idx+window_size-1]['close']) - 1
                
                X_support.append(window.flatten())
                y_support.append(target)
            
            for idx in query_indices:
                window = asset_data.iloc[idx:idx+window_size].values
                
                if self.task_type == 'classification':
                    target = 1 if asset_data.iloc[idx+window_size]['close'] > asset_data.iloc[idx+window_size-1]['close'] else 0
                else:
                    target = (asset_data.iloc[idx+window_size]['close'] / asset_data.iloc[idx+window_size-1]['close']) - 1
                
                X_query.append(window.flatten())
                y_query.append(target)
            
            if len(X_support) > 0 and len(X_query) > 0:
                X_support = torch.FloatTensor(X_support)
                y_support = torch.LongTensor(y_support) if self.task_type == 'classification' else torch.FloatTensor(y_support)
                X_query = torch.FloatTensor(X_query)
                y_query = torch.LongTensor(y_query) if self.task_type == 'classification' else torch.FloatTensor(y_query)
                
                support_sets.append((X_support, y_support))
                query_sets.append((X_query, y_query))
        
        return support_sets, query_sets
    
    def meta_train_on_financial_data(
        self,
        financial_data: Dict[str, pd.DataFrame],
        n_epochs: int = 100,
        window_size: int = 20,
        n_tasks: int = 50
    ) -> Dict[str, float]:
        """Meta-train on financial data"""
        
        # Create tasks from financial data
        support_sets, query_sets = self._create_financial_tasks(
            financial_data, window_size, n_tasks
        )
        
        print(f"Created {len(support_sets)} financial tasks")
        
        # Meta-train
        return self.base_learner.meta_train(support_sets, query_sets, n_epochs)
    
    def adapt_to_regime(
        self,
        recent_data: pd.DataFrame,
        window_size: int = 20,
        n_support: int = 10,
        n_adaptation_steps: int = 10
    ) -> None:
        """
        Quickly adapt to new market regime
        
        Args:
            recent_data: Recent market data
            window_size: Size of input windows
            n_support: Number of support examples
            n_adaptation_steps: Number of adaptation steps
        """
        
        if len(recent_data) < window_size + n_support:
            raise ValueError("Not enough data for adaptation")
        
        # Create support set from recent data
        X_support = []
        y_support = []
        
        for i in range(n_support):
            start_idx = len(recent_data) - window_size - n_support + i
            window = recent_data.iloc[start_idx:start_idx+window_size].values
            
            if self.task_type == 'classification':
                target = 1 if recent_data.iloc[start_idx+window_size]['close'] > recent_data.iloc[start_idx+window_size-1]['close'] else 0
            else:
                target = (recent_data.iloc[start_idx+window_size]['close'] / recent_data.iloc[start_idx+window_size-1]['close']) - 1
            
            X_support.append(window.flatten())
            y_support.append(target)
        
        X_support = torch.FloatTensor(X_support)
        y_support = torch.LongTensor(y_support) if self.task_type == 'classification' else torch.FloatTensor(y_support)
        
        # Adapt
        self.base_learner.adapt(X_support, y_support, n_adaptation_steps)
        
        # Store regime info
        self.regime_history.append({
            'timestamp': recent_data.index[-1],
            'volatility': recent_data['close'].pct_change().std(),
            'trend': recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1
        })
    
    def predict_next_period(
        self,
        current_window: pd.DataFrame
    ) -> Union[int, float]:
        """
        Predict next period outcome
        
        Args:
            current_window: Current market window
        
        Returns:
            Prediction (class or value)
        """
        
        X = torch.FloatTensor(current_window.values.flatten()).unsqueeze(0)
        X = self._extract_features(X)
        
        prediction = self.base_learner.predict(X)
        
        return prediction.item()


def evaluate_few_shot_performance(
    learner: FewShotLearner,
    test_tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    n_adaptation_steps: int = 10
) -> Dict[str, float]:
    """
    Evaluate few-shot learning performance
    
    Args:
        learner: Few-shot learner
        test_tasks: List of (X_support, y_support, X_query, y_query) tuples
        n_adaptation_steps: Number of adaptation steps
    
    Returns:
        Performance metrics
    """
    
    accuracies = []
    mse_scores = []
    
    for X_support, y_support, X_query, y_query in test_tasks:
        
        # Adapt to task
        learner.adapt(X_support, y_support, n_adaptation_steps)
        
        # Make predictions
        predictions = learner.predict(X_query)
        
        if y_query.dtype == torch.long:
            # Classification
            accuracy = accuracy_score(y_query.numpy(), predictions.numpy())
            accuracies.append(accuracy)
        else:
            # Regression
            mse = mean_squared_error(y_query.numpy(), predictions.numpy())
            mse_scores.append(mse)
    
    results = {}
    if accuracies:
        results['mean_accuracy'] = np.mean(accuracies)
        results['std_accuracy'] = np.std(accuracies)
    
    if mse_scores:
        results['mean_mse'] = np.mean(mse_scores)
        results['std_mse'] = np.std(mse_scores)
    
    return results


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Testing Few-Shot Learning Methods...")
    
    # Create simple MLP for testing
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.output(x)
    
    # Test with synthetic data
    input_dim = 10
    n_classes = 5
    n_tasks = 20
    
    # Generate synthetic tasks
    support_sets = []
    query_sets = []
    
    for task in range(n_tasks):
        # Random linear transformation for each task
        W = torch.randn(input_dim, input_dim) * 0.5
        b = torch.randn(input_dim) * 0.1
        
        # Support set
        X_support = torch.randn(25, input_dim)  # 5 examples per class
        X_support = torch.mm(X_support, W) + b
        y_support = torch.randint(0, n_classes, (25,))
        
        # Query set
        X_query = torch.randn(25, input_dim)
        X_query = torch.mm(X_query, W) + b
        y_query = torch.randint(0, n_classes, (25,))
        
        support_sets.append((X_support, y_support))
        query_sets.append((X_query, y_query))
    
    # Test MAML
    print("\nTesting MAML...")
    try:
        model = SimpleMLP(input_dim, 64, n_classes)
        maml = MAML(model, inner_lr=0.01, meta_lr=0.001, inner_steps=5)
        
        # Meta-train
        maml_results = maml.meta_train(support_sets[:15], query_sets[:15], n_epochs=50)
        print(f"MAML meta-training completed")
        
        # Test adaptation
        test_support, test_query = support_sets[15], query_sets[15]
        maml.adapt(test_support[0], test_support[1], n_steps=10)
        
        predictions = maml.predict(test_query[0])
        accuracy = accuracy_score(test_query[1].numpy(), predictions.argmax(dim=1).numpy())
        print(f"MAML Test Accuracy: {accuracy:.3f}")
        
    except Exception as e:
        print(f"MAML error: {e}")
    
    # Test Prototypical Networks
    print("\nTesting Prototypical Networks...")
    try:
        encoder = SimpleMLP(input_dim, 64, 32)  # Output embedding dim = 32
        protonet = ProtoNet(encoder, distance_metric='euclidean')
        
        # Meta-train
        proto_results = protonet.meta_train(support_sets[:15], query_sets[:15], n_epochs=50)
        print(f"ProtoNet meta-training completed")
        
        # Test adaptation
        protonet.adapt(test_support[0], test_support[1])
        predictions = protonet.predict(test_query[0])
        accuracy = accuracy_score(test_query[1].numpy(), predictions.numpy())
        print(f"ProtoNet Test Accuracy: {accuracy:.3f}")
        
    except Exception as e:
        print(f"ProtoNet error: {e}")
    
    # Test with financial-like data
    print("\nTesting Financial Few-Shot Learning...")
    try:
        # Generate synthetic financial data
        n_assets = 5
        n_periods = 500
        
        financial_data = {}
        
        for asset_id in range(n_assets):
            # Random walk with different parameters for each asset
            returns = np.random.normal(0.001, 0.02, n_periods)  # Daily returns
            prices = 100 * np.cumprod(1 + returns)
            
            # Create OHLCV data
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_periods))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_periods))),
                'close': prices,
                'volume': np.random.exponential(1000000, n_periods)
            })
            
            financial_data[f'Asset_{asset_id}'] = df
        
        # Create financial few-shot learner
        base_model = SimpleMLP(100, 128, 2)  # 20 periods * 5 features = 100 input dim
        maml_financial = MAML(base_model, inner_lr=0.01, meta_lr=0.001)
        
        financial_learner = FinancialFewShotLearner(
            base_learner=maml_financial,
            task_type='classification'
        )
        
        # Meta-train on financial data
        results = financial_learner.meta_train_on_financial_data(
            financial_data, n_epochs=30, window_size=20, n_tasks=20
        )
        
        print("Financial meta-training completed")
        
        # Test regime adaptation
        test_data = financial_data['Asset_0'].iloc[-50:]  # Recent data
        financial_learner.adapt_to_regime(test_data, window_size=20, n_support=10)
        
        # Make prediction
        current_window = test_data.iloc[-20:]
        prediction = financial_learner.predict_next_period(current_window)
        print(f"Financial prediction: {'Up' if prediction == 1 else 'Down'}")
        
    except Exception as e:
        print(f"Financial few-shot learning error: {e}")
    
    print("\nDone!")