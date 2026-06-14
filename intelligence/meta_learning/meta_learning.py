"""
Meta-Learning Systems for QUANTUM-FORGE
Implements advanced meta-learning algorithms including MAML, Prototypical Networks,
Relation Networks, and few-shot learning for adaptive trading strategies.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import warnings
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import copy
import random
import pickle
warnings.filterwarnings('ignore')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetaLearningType(Enum):
    """Types of meta-learning algorithms."""
    MAML = "model_agnostic_meta_learning"
    FOMAML = "first_order_maml"
    REPTILE = "reptile"
    PROTOTYPICAL = "prototypical_networks"
    RELATION = "relation_networks"
    MATCHING = "matching_networks"
    SNAIL = "simple_neural_attentive_learner"
    META_SGD = "meta_sgd"

class TaskType(Enum):
    """Types of meta-learning tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TRADING_STRATEGY = "trading_strategy"

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning algorithms."""
    algorithm_type: MetaLearningType
    task_type: TaskType
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    meta_learning_rate: float
    inner_learning_rate: float
    num_inner_steps: int
    num_meta_epochs: int
    meta_batch_size: int
    support_size: int  # N-way K-shot: K
    query_size: int
    num_ways: int     # N-way K-shot: N

class Task:
    """Individual task for meta-learning."""
    
    def __init__(self, task_id: str, support_data: torch.Tensor, support_labels: torch.Tensor,
                 query_data: torch.Tensor, query_labels: torch.Tensor, task_info: Dict = None):
        """Initialize task."""
        self.task_id = task_id
        self.support_data = support_data
        self.support_labels = support_labels
        self.query_data = query_data
        self.query_labels = query_labels
        self.task_info = task_info or {}
    
    def to_device(self, device):
        """Move task data to device."""
        self.support_data = self.support_data.to(device)
        self.support_labels = self.support_labels.to(device)
        self.query_data = self.query_data.to(device)
        self.query_labels = self.query_labels.to(device)
        return self

class MetaModel(nn.Module):
    """Base meta-learning model."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int):
        """Initialize meta-model."""
        super(MetaModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build network layers
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation for last layer
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)
    
    def get_parameters(self):
        """Get model parameters as ordered dictionary."""
        return OrderedDict(self.named_parameters())
    
    def set_parameters(self, params):
        """Set model parameters from ordered dictionary."""
        for name, param in self.named_parameters():
            param.data.copy_(params[name].data)
    
    def copy(self):
        """Create a copy of the model."""
        new_model = type(self)(self.input_dim, self.output_dim, self.hidden_dim, self.num_layers)
        new_model.load_state_dict(self.state_dict())
        return new_model

class MAML:
    """Model-Agnostic Meta-Learning algorithm."""
    
    def __init__(self, model: MetaModel, config: MetaLearningConfig):
        """Initialize MAML."""
        self.model = model.to(DEVICE)
        self.config = config
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=config.meta_learning_rate)
        
        # Training history
        self.training_history = []
        
    def inner_loop(self, task: Task) -> Tuple[torch.Tensor, Dict]:
        """Perform inner loop adaptation for a single task."""
        # Create a copy of the model for inner loop
        adapted_model = self.model.copy().to(DEVICE)
        adapted_params = adapted_model.get_parameters()
        
        # Inner loop optimizer
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.inner_learning_rate)
        
        # Adapt to support set
        for step in range(self.config.num_inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass on support set
            support_pred = adapted_model(task.support_data)
            
            # Compute loss
            if self.config.task_type == TaskType.CLASSIFICATION:
                support_loss = F.cross_entropy(support_pred, task.support_labels.long())
            else:  # Regression
                support_loss = F.mse_loss(support_pred, task.support_labels)
            
            # Backward pass
            support_loss.backward()
            inner_optimizer.step()
        
        # Evaluate on query set
        query_pred = adapted_model(task.query_data)
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            query_loss = F.cross_entropy(query_pred, task.query_labels.long())
            accuracy = (query_pred.argmax(dim=1) == task.query_labels.long()).float().mean()
            metrics = {'accuracy': accuracy.item()}
        else:
            query_loss = F.mse_loss(query_pred, task.query_labels)
            metrics = {'mse': query_loss.item()}
        
        return query_loss, metrics
    
    def train_step(self, task_batch: List[Task]) -> Dict:
        """Perform one meta-training step."""
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        batch_metrics = defaultdict(list)
        
        # Process each task in the batch
        for task in task_batch:
            task = task.to_device(DEVICE)
            
            # Inner loop adaptation
            query_loss, task_metrics = self.inner_loop(task)
            
            # Accumulate loss for meta-update
            total_loss += query_loss
            
            # Collect metrics
            for key, value in task_metrics.items():
                batch_metrics[key].append(value)
        
        # Meta-update
        avg_loss = total_loss / len(task_batch)
        avg_loss.backward()
        self.meta_optimizer.step()
        
        # Average metrics
        averaged_metrics = {key: np.mean(values) for key, values in batch_metrics.items()}
        averaged_metrics['meta_loss'] = avg_loss.item()
        
        return averaged_metrics
    
    def evaluate(self, task_batch: List[Task]) -> Dict:
        """Evaluate on a batch of tasks."""
        self.model.eval()
        
        batch_metrics = defaultdict(list)
        
        with torch.no_grad():
            for task in task_batch:
                task = task.to_device(DEVICE)
                
                # Inner loop adaptation (no gradient updates to meta-parameters)
                query_loss, task_metrics = self.inner_loop(task)
                
                # Collect metrics
                for key, value in task_metrics.items():
                    batch_metrics[key].append(value)
        
        self.model.train()
        
        # Average metrics
        averaged_metrics = {key: np.mean(values) for key, values in batch_metrics.items()}
        
        return averaged_metrics

class Reptile:
    """Reptile meta-learning algorithm."""
    
    def __init__(self, model: MetaModel, config: MetaLearningConfig):
        """Initialize Reptile."""
        self.model = model.to(DEVICE)
        self.config = config
        
        self.training_history = []
    
    def train_step(self, task_batch: List[Task]) -> Dict:
        """Perform one Reptile training step."""
        # Store initial parameters
        initial_params = OrderedDict()
        for name, param in self.model.named_parameters():
            initial_params[name] = param.clone()
        
        batch_metrics = defaultdict(list)
        
        # Process each task in the batch
        for task in task_batch:
            task = task.to_device(DEVICE)
            
            # Reset to initial parameters
            self.model.set_parameters(initial_params)
            
            # Create optimizer for this task
            optimizer = optim.SGD(self.model.parameters(), lr=self.config.inner_learning_rate)
            
            # Adapt to task
            for step in range(self.config.num_inner_steps):
                optimizer.zero_grad()
                
                # Use both support and query data for Reptile
                all_data = torch.cat([task.support_data, task.query_data], dim=0)
                all_labels = torch.cat([task.support_labels, task.query_labels], dim=0)
                
                pred = self.model(all_data)
                
                if self.config.task_type == TaskType.CLASSIFICATION:
                    loss = F.cross_entropy(pred, all_labels.long())
                else:
                    loss = F.mse_loss(pred, all_labels)
                
                loss.backward()
                optimizer.step()
            
            # Evaluate on query set
            with torch.no_grad():
                query_pred = self.model(task.query_data)
                
                if self.config.task_type == TaskType.CLASSIFICATION:
                    query_loss = F.cross_entropy(query_pred, task.query_labels.long())
                    accuracy = (query_pred.argmax(dim=1) == task.query_labels.long()).float().mean()
                    batch_metrics['accuracy'].append(accuracy.item())
                else:
                    query_loss = F.mse_loss(query_pred, task.query_labels)
                    batch_metrics['mse'].append(query_loss.item())
        
        # Reptile update: move towards the average of adapted parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Move towards adapted parameters
                param.data = initial_params[name] + self.config.meta_learning_rate * (
                    param.data - initial_params[name]
                )
        
        # Average metrics
        averaged_metrics = {key: np.mean(values) for key, values in batch_metrics.items()}
        
        return averaged_metrics

class PrototypicalNetworks(nn.Module):
    """Prototypical Networks for few-shot learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        """Initialize Prototypical Networks."""
        super(PrototypicalNetworks, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Embedding network
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        self.embedding_net = nn.Sequential(*layers)
        
    def forward(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                query_data: torch.Tensor, num_ways: int) -> torch.Tensor:
        """Forward pass for Prototypical Networks."""
        
        # Embed support and query data
        support_embeddings = self.embedding_net(support_data)
        query_embeddings = self.embedding_net(query_data)
        
        # Compute prototypes (class centers)
        prototypes = []
        for way in range(num_ways):
            class_mask = (support_labels == way)
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute distances to prototypes
        distances = torch.cdist(query_embeddings, prototypes)
        
        # Convert distances to logits (negative distances)
        logits = -distances
        
        return logits

class RelationNetworks(nn.Module):
    """Relation Networks for few-shot learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, relation_dim: int = 8):
        """Initialize Relation Networks."""
        super(RelationNetworks, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim
        
        # Feature embedding network
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Relation network
        self.relation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                query_data: torch.Tensor, num_ways: int) -> torch.Tensor:
        """Forward pass for Relation Networks."""
        
        # Embed support and query data
        support_features = self.feature_net(support_data)
        query_features = self.feature_net(query_data)
        
        # Compute class prototypes
        prototypes = []
        for way in range(num_ways):
            class_mask = (support_labels == way)
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute relations
        num_query = query_features.shape[0]
        relations = []
        
        for i in range(num_query):
            query_feature = query_features[i].unsqueeze(0).repeat(num_ways, 1)
            
            # Concatenate query feature with each prototype
            combined_features = torch.cat([prototypes, query_feature], dim=1)
            
            # Compute relation scores
            relation_scores = self.relation_net(combined_features).squeeze()
            relations.append(relation_scores)
        
        relations = torch.stack(relations)
        
        return relations

class MatchingNetworks(nn.Module):
    """Matching Networks for few-shot learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        """Initialize Matching Networks."""
        super(MatchingNetworks, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder for support set
        self.support_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder for query set
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                query_data: torch.Tensor, num_ways: int) -> torch.Tensor:
        """Forward pass for Matching Networks."""
        
        # Encode support and query sets
        support_encoded = self.support_encoder(support_data)
        query_encoded = self.query_encoder(query_data)
        
        # Apply attention
        attended_query, attention_weights = self.attention(
            query_encoded.unsqueeze(0), 
            support_encoded.unsqueeze(0), 
            support_encoded.unsqueeze(0)
        )
        attended_query = attended_query.squeeze(0)
        
        # Compute similarities
        similarities = torch.matmul(attended_query, support_encoded.T)
        similarities = F.softmax(similarities, dim=1)
        
        # Weighted combination of support labels
        support_labels_one_hot = F.one_hot(support_labels.long(), num_classes=num_ways).float()
        logits = torch.matmul(similarities, support_labels_one_hot)
        
        return logits

class TaskGenerator:
    """Generate tasks for meta-learning."""
    
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, num_ways: int,
                 support_size: int, query_size: int):
        """Initialize task generator."""
        self.data = data
        self.labels = labels
        self.num_ways = num_ways
        self.support_size = support_size
        self.query_size = query_size
        
        # Group samples by class
        self.class_data = {}
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            mask = (labels == label)
            self.class_data[label.item()] = data[mask]
    
    def generate_task(self, task_id: str = None) -> Task:
        """Generate a single task."""
        if task_id is None:
            task_id = f"task_{random.randint(0, 9999)}"
        
        # Randomly select classes for this task
        available_classes = list(self.class_data.keys())
        selected_classes = random.sample(available_classes, self.num_ways)
        
        support_data_list = []
        support_labels_list = []
        query_data_list = []
        query_labels_list = []
        
        for new_label, original_class in enumerate(selected_classes):
            class_data = self.class_data[original_class]
            
            # Randomly sample from this class
            num_samples = len(class_data)
            total_needed = self.support_size + self.query_size
            
            if num_samples < total_needed:
                # If not enough samples, sample with replacement
                indices = torch.randint(0, num_samples, (total_needed,))
            else:
                indices = torch.randperm(num_samples)[:total_needed]
            
            sampled_data = class_data[indices]
            
            # Split into support and query
            support_data_list.append(sampled_data[:self.support_size])
            query_data_list.append(sampled_data[self.support_size:self.support_size + self.query_size])
            
            support_labels_list.extend([new_label] * self.support_size)
            query_labels_list.extend([new_label] * self.query_size)
        
        # Combine data
        support_data = torch.cat(support_data_list, dim=0)
        support_labels = torch.tensor(support_labels_list)
        query_data = torch.cat(query_data_list, dim=0)
        query_labels = torch.tensor(query_labels_list)
        
        # Shuffle
        support_perm = torch.randperm(len(support_data))
        support_data = support_data[support_perm]
        support_labels = support_labels[support_perm]
        
        query_perm = torch.randperm(len(query_data))
        query_data = query_data[query_perm]
        query_labels = query_labels[query_perm]
        
        return Task(task_id, support_data, support_labels, query_data, query_labels)
    
    def generate_batch(self, batch_size: int) -> List[Task]:
        """Generate a batch of tasks."""
        return [self.generate_task() for _ in range(batch_size)]

class MetaLearningEngine:
    """Main engine for meta-learning algorithms."""
    
    def __init__(self):
        """Initialize meta-learning engine."""
        self.algorithms = {}
        self.task_generators = {}
        self.evaluation_results = {}
        self.device = DEVICE
    
    def create_maml(self, algorithm_name: str, config: MetaLearningConfig) -> MAML:
        """Create and register MAML algorithm."""
        model = MetaModel(config.input_dim, config.output_dim, 
                         config.hidden_dim, config.num_layers)
        
        if config.algorithm_type == MetaLearningType.MAML:
            algorithm = MAML(model, config)
        elif config.algorithm_type == MetaLearningType.REPTILE:
            algorithm = Reptile(model, config)
        else:
            raise ValueError(f"Unsupported algorithm type: {config.algorithm_type}")
        
        self.algorithms[algorithm_name] = algorithm
        return algorithm
    
    def create_prototypical_net(self, algorithm_name: str, input_dim: int, 
                              hidden_dim: int, num_layers: int = 3) -> PrototypicalNetworks:
        """Create and register Prototypical Networks."""
        model = PrototypicalNetworks(input_dim, hidden_dim, num_layers).to(self.device)
        self.algorithms[algorithm_name] = model
        return model
    
    def create_relation_net(self, algorithm_name: str, input_dim: int, 
                          hidden_dim: int, relation_dim: int = 8) -> RelationNetworks:
        """Create and register Relation Networks."""
        model = RelationNetworks(input_dim, hidden_dim, relation_dim).to(self.device)
        self.algorithms[algorithm_name] = model
        return model
    
    def create_matching_net(self, algorithm_name: str, input_dim: int, 
                          hidden_dim: int, num_layers: int = 2) -> MatchingNetworks:
        """Create and register Matching Networks."""
        model = MatchingNetworks(input_dim, hidden_dim, num_layers).to(self.device)
        self.algorithms[algorithm_name] = model
        return model
    
    def create_task_generator(self, generator_name: str, data: torch.Tensor, 
                            labels: torch.Tensor, num_ways: int, support_size: int,
                            query_size: int) -> TaskGenerator:
        """Create and register task generator."""
        generator = TaskGenerator(data, labels, num_ways, support_size, query_size)
        self.task_generators[generator_name] = generator
        return generator
    
    def train_gradient_based(self, algorithm_name: str, generator_name: str,
                           num_epochs: int, batch_size: int) -> Dict:
        """Train gradient-based meta-learning algorithms (MAML, Reptile)."""
        
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        if generator_name not in self.task_generators:
            raise ValueError(f"Task generator {generator_name} not found")
        
        algorithm = self.algorithms[algorithm_name]
        generator = self.task_generators[generator_name]
        
        training_history = []
        
        for epoch in range(num_epochs):
            # Generate batch of tasks
            task_batch = generator.generate_batch(batch_size)
            
            # Training step
            metrics = algorithm.train_step(task_batch)
            
            epoch_info = {
                'epoch': epoch,
                **metrics
            }
            training_history.append(epoch_info)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: {metrics}")
        
        algorithm.training_history = training_history
        return training_history
    
    def train_metric_based(self, algorithm_name: str, generator_name: str,
                         num_epochs: int, batch_size: int, learning_rate: float = 0.001) -> Dict:
        """Train metric-based meta-learning algorithms (Prototypical, Relation, Matching Networks)."""
        
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        if generator_name not in self.task_generators:
            raise ValueError(f"Task generator {generator_name} not found")
        
        model = self.algorithms[algorithm_name]
        generator = self.task_generators[generator_name]
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        training_history = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for batch_idx in range(batch_size):
                # Generate task
                task = generator.generate_task()
                task = task.to_device(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(task.support_data, task.support_labels, 
                             task.query_data, generator.num_ways)
                
                # Compute loss
                if isinstance(model, RelationNetworks):
                    # For relation networks, labels are probabilities
                    target_relations = F.one_hot(task.query_labels.long(), 
                                               num_classes=generator.num_ways).float()
                    loss = F.binary_cross_entropy(logits, target_relations)
                else:
                    loss = F.cross_entropy(logits, task.query_labels.long())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Compute accuracy
                predicted = logits.argmax(dim=1)
                accuracy = (predicted == task.query_labels.long()).float().mean()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            epoch_info = {
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_accuracy
            }
            training_history.append(epoch_info)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.6f}, Accuracy={avg_accuracy:.4f}")
        
        return training_history
    
    def evaluate_algorithm(self, algorithm_name: str, generator_name: str,
                         num_tasks: int = 100) -> Dict:
        """Evaluate meta-learning algorithm."""
        
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        if generator_name not in self.task_generators:
            raise ValueError(f"Task generator {generator_name} not found")
        
        algorithm = self.algorithms[algorithm_name]
        generator = self.task_generators[generator_name]
        
        # Generate evaluation tasks
        eval_tasks = generator.generate_batch(num_tasks)
        
        if isinstance(algorithm, (MAML, Reptile)):
            # Gradient-based evaluation
            results = algorithm.evaluate(eval_tasks)
        else:
            # Metric-based evaluation
            total_loss = 0.0
            total_accuracy = 0.0
            
            algorithm.eval()
            with torch.no_grad():
                for task in eval_tasks:
                    task = task.to_device(self.device)
                    
                    logits = algorithm(task.support_data, task.support_labels,
                                     task.query_data, generator.num_ways)
                    
                    if isinstance(algorithm, RelationNetworks):
                        target_relations = F.one_hot(task.query_labels.long(),
                                                   num_classes=generator.num_ways).float()
                        loss = F.binary_cross_entropy(logits, target_relations)
                    else:
                        loss = F.cross_entropy(logits, task.query_labels.long())
                    
                    predicted = logits.argmax(dim=1)
                    accuracy = (predicted == task.query_labels.long()).float().mean()
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
            
            algorithm.train()
            
            results = {
                'loss': total_loss / num_tasks,
                'accuracy': total_accuracy / num_tasks
            }
        
        self.evaluation_results[algorithm_name] = results
        return results
    
    def few_shot_adaptation(self, algorithm_name: str, new_task: Task,
                          adaptation_steps: int = 5) -> Dict:
        """Perform few-shot adaptation on a new task."""
        
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        algorithm = self.algorithms[algorithm_name]
        new_task = new_task.to_device(self.device)
        
        if isinstance(algorithm, (MAML, Reptile)):
            # Create adapted model
            adapted_model = algorithm.model.copy().to(self.device)
            optimizer = optim.SGD(adapted_model.parameters(), 
                                lr=algorithm.config.inner_learning_rate)
            
            # Adaptation on support set
            adaptation_history = []
            
            for step in range(adaptation_steps):
                optimizer.zero_grad()
                
                support_pred = adapted_model(new_task.support_data)
                
                if algorithm.config.task_type == TaskType.CLASSIFICATION:
                    loss = F.cross_entropy(support_pred, new_task.support_labels.long())
                else:
                    loss = F.mse_loss(support_pred, new_task.support_labels)
                
                loss.backward()
                optimizer.step()
                
                adaptation_history.append({
                    'step': step,
                    'support_loss': loss.item()
                })
            
            # Evaluate on query set
            with torch.no_grad():
                query_pred = adapted_model(new_task.query_data)
                
                if algorithm.config.task_type == TaskType.CLASSIFICATION:
                    query_loss = F.cross_entropy(query_pred, new_task.query_labels.long())
                    accuracy = (query_pred.argmax(dim=1) == new_task.query_labels.long()).float().mean()
                    final_metrics = {'query_loss': query_loss.item(), 'accuracy': accuracy.item()}
                else:
                    query_loss = F.mse_loss(query_pred, new_task.query_labels)
                    final_metrics = {'query_loss': query_loss.item()}
            
            return {
                'adaptation_history': adaptation_history,
                'final_metrics': final_metrics,
                'adapted_model': adapted_model
            }
        
        else:
            # Metric-based algorithms don't need explicit adaptation
            with torch.no_grad():
                logits = algorithm(new_task.support_data, new_task.support_labels,
                                 new_task.query_data, len(torch.unique(new_task.support_labels)))
                
                predicted = logits.argmax(dim=1)
                accuracy = (predicted == new_task.query_labels.long()).float().mean()
                
                if isinstance(algorithm, RelationNetworks):
                    target_relations = F.one_hot(new_task.query_labels.long(),
                                               num_classes=len(torch.unique(new_task.support_labels))).float()
                    loss = F.binary_cross_entropy(logits, target_relations)
                else:
                    loss = F.cross_entropy(logits, new_task.query_labels.long())
            
            return {
                'final_metrics': {
                    'query_loss': loss.item(),
                    'accuracy': accuracy.item()
                }
            }
    
    def get_algorithm_summary(self, algorithm_name: str) -> Dict:
        """Get algorithm summary."""
        
        if algorithm_name not in self.algorithms:
            return {}
        
        algorithm = self.algorithms[algorithm_name]
        
        summary = {
            'algorithm_name': algorithm_name,
            'algorithm_type': type(algorithm).__name__
        }
        
        if isinstance(algorithm, (MAML, Reptile)):
            summary.update({
                'model_parameters': sum(p.numel() for p in algorithm.model.parameters()),
                'input_dim': algorithm.model.input_dim,
                'output_dim': algorithm.model.output_dim,
                'hidden_dim': algorithm.model.hidden_dim,
                'meta_learning_rate': algorithm.config.meta_learning_rate,
                'inner_learning_rate': algorithm.config.inner_learning_rate,
                'num_inner_steps': algorithm.config.num_inner_steps
            })
            
            if hasattr(algorithm, 'training_history') and algorithm.training_history:
                summary['training_epochs'] = len(algorithm.training_history)
                summary['final_metrics'] = algorithm.training_history[-1]
        
        else:
            summary.update({
                'model_parameters': sum(p.numel() for p in algorithm.parameters()),
                'input_dim': algorithm.input_dim,
                'hidden_dim': algorithm.hidden_dim
            })
        
        # Add evaluation results if available
        if algorithm_name in self.evaluation_results:
            summary['evaluation_results'] = self.evaluation_results[algorithm_name]
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    print("Testing Meta-Learning Systems...")
    
    # Generate deterministic multi-class data for few-shot learning (no randomness)
    # Create 10 classes of data
    num_classes = 10
    samples_per_class = 100
    feature_dim = 20

    all_data = []
    all_labels = []

    for class_id in range(num_classes):
        # Deterministic class mean that shifts with class_id
        class_mean = torch.linspace(-1.0, 1.0, feature_dim) + class_id * 0.1
        # Deterministic per-sample variation using a linspace noise pattern
        noise = torch.linspace(-0.1, 0.1, samples_per_class).unsqueeze(1).repeat(1, feature_dim)
        class_data = class_mean.unsqueeze(0).repeat(samples_per_class, 1) + noise

        all_data.append(class_data)
        all_labels.extend([class_id] * samples_per_class)
    
    X = torch.cat(all_data, dim=0)
    y = torch.tensor(all_labels)
    
    # Initialize meta-learning engine
    engine = MetaLearningEngine()
    
    # Create task generator (5-way 5-shot learning)
    num_ways = 5
    support_size = 5  # K-shot
    query_size = 15
    
    task_gen = engine.create_task_generator(
        "few_shot_tasks", X, y, num_ways, support_size, query_size
    )
    
    # Test MAML
    print("\n=== Testing MAML ===")
    
    maml_config = MetaLearningConfig(
        algorithm_type=MetaLearningType.MAML,
        task_type=TaskType.CLASSIFICATION,
        input_dim=feature_dim,
        output_dim=num_ways,
        hidden_dim=64,
        num_layers=2,
        meta_learning_rate=0.001,
        inner_learning_rate=0.01,
        num_inner_steps=5,
        num_meta_epochs=100,
        meta_batch_size=4,
        support_size=support_size,
        query_size=query_size,
        num_ways=num_ways
    )
    
    maml = engine.create_maml("maml_classifier", maml_config)
    
    # Train MAML
    print("Training MAML...")
    maml_history = engine.train_gradient_based("maml_classifier", "few_shot_tasks", 
                                              num_epochs=50, batch_size=4)
    
    # Evaluate MAML
    maml_results = engine.evaluate_algorithm("maml_classifier", "few_shot_tasks", num_tasks=50)
    print(f"MAML Evaluation - Accuracy: {maml_results['accuracy']:.4f}")
    
    # Test Reptile
    print("\n=== Testing Reptile ===")
    
    reptile_config = MetaLearningConfig(
        algorithm_type=MetaLearningType.REPTILE,
        task_type=TaskType.CLASSIFICATION,
        input_dim=feature_dim,
        output_dim=num_ways,
        hidden_dim=64,
        num_layers=2,
        meta_learning_rate=0.001,
        inner_learning_rate=0.01,
        num_inner_steps=10,
        num_meta_epochs=100,
        meta_batch_size=4,
        support_size=support_size,
        query_size=query_size,
        num_ways=num_ways
    )
    
    reptile = engine.create_maml("reptile_classifier", reptile_config)
    
    # Train Reptile
    print("Training Reptile...")
    reptile_history = engine.train_gradient_based("reptile_classifier", "few_shot_tasks",
                                                 num_epochs=50, batch_size=4)
    
    # Evaluate Reptile
    reptile_results = engine.evaluate_algorithm("reptile_classifier", "few_shot_tasks", num_tasks=50)
    print(f"Reptile Evaluation - Accuracy: {reptile_results['accuracy']:.4f}")
    
    # Test Prototypical Networks
    print("\n=== Testing Prototypical Networks ===")
    
    proto_net = engine.create_prototypical_net("proto_net", feature_dim, 64, num_layers=3)
    
    # Train Prototypical Networks
    print("Training Prototypical Networks...")
    proto_history = engine.train_metric_based("proto_net", "few_shot_tasks",
                                             num_epochs=100, batch_size=32, learning_rate=0.001)
    
    # Evaluate Prototypical Networks
    proto_results = engine.evaluate_algorithm("proto_net", "few_shot_tasks", num_tasks=50)
    print(f"Prototypical Networks Evaluation - Accuracy: {proto_results['accuracy']:.4f}")
    
    # Test Relation Networks
    print("\n=== Testing Relation Networks ===")
    
    relation_net = engine.create_relation_net("relation_net", feature_dim, 64, relation_dim=8)
    
    # Train Relation Networks
    print("Training Relation Networks...")
    relation_history = engine.train_metric_based("relation_net", "few_shot_tasks",
                                                num_epochs=100, batch_size=32, learning_rate=0.001)
    
    # Evaluate Relation Networks
    relation_results = engine.evaluate_algorithm("relation_net", "few_shot_tasks", num_tasks=50)
    print(f"Relation Networks Evaluation - Accuracy: {relation_results['accuracy']:.4f}")
    
    # Test Matching Networks
    print("\n=== Testing Matching Networks ===")
    
    matching_net = engine.create_matching_net("matching_net", feature_dim, 64, num_layers=2)
    
    # Train Matching Networks
    print("Training Matching Networks...")
    matching_history = engine.train_metric_based("matching_net", "few_shot_tasks",
                                                num_epochs=100, batch_size=32, learning_rate=0.001)
    
    # Evaluate Matching Networks
    matching_results = engine.evaluate_algorithm("matching_net", "few_shot_tasks", num_tasks=50)
    print(f"Matching Networks Evaluation - Accuracy: {matching_results['accuracy']:.4f}")
    
    # Test few-shot adaptation
    print("\n=== Testing Few-Shot Adaptation ===")
    
    # Generate a new task for adaptation
    new_task = task_gen.generate_task("adaptation_task")
    
    # Test adaptation with MAML
    maml_adaptation = engine.few_shot_adaptation("maml_classifier", new_task, adaptation_steps=10)
    print(f"MAML Adaptation - Final Accuracy: {maml_adaptation['final_metrics']['accuracy']:.4f}")
    
    # Test adaptation with Prototypical Networks
    proto_adaptation = engine.few_shot_adaptation("proto_net", new_task)
    print(f"Prototypical Networks Adaptation - Accuracy: {proto_adaptation['final_metrics']['accuracy']:.4f}")
    
    # Algorithm summaries
    print("\n=== Algorithm Summaries ===")
    
    algorithms = ["maml_classifier", "reptile_classifier", "proto_net", "relation_net", "matching_net"]
    
    for alg_name in algorithms:
        summary = engine.get_algorithm_summary(alg_name)
        print(f"\n{alg_name}:")
        for key, value in summary.items():
            if key not in ['algorithm_name', 'final_metrics', 'evaluation_results']:
                print(f"  {key}: {value}")
        
        if 'evaluation_results' in summary:
            eval_results = summary['evaluation_results']
            print(f"  evaluation_accuracy: {eval_results.get('accuracy', 'N/A')}")
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    print(f"{'Algorithm':<20} {'Accuracy':<10} {'Parameters':<12}")
    print("-" * 45)
    
    for alg_name in algorithms:
        summary = engine.get_algorithm_summary(alg_name)
        accuracy = summary.get('evaluation_results', {}).get('accuracy', 0.0)
        params = summary.get('model_parameters', 0)
        
        print(f"{alg_name:<20} {accuracy:<10.4f} {params:<12}")
    
    # Test different shot scenarios
    print("\n=== Testing Different Shot Scenarios ===")
    
    shot_scenarios = [1, 3, 5, 10]
    
    for k_shot in shot_scenarios:
        print(f"\n{k_shot}-shot learning:")
        
        # Create new task generator for this scenario
        shot_gen = engine.create_task_generator(
            f"{k_shot}_shot_tasks", X, y, num_ways, k_shot, query_size
        )
        
        # Test with MAML (quick evaluation)
        eval_tasks = shot_gen.generate_batch(20)
        if isinstance(engine.algorithms["maml_classifier"], MAML):
            results = engine.algorithms["maml_classifier"].evaluate(eval_tasks)
            print(f"  MAML {k_shot}-shot accuracy: {results.get('accuracy', 0.0):.4f}")
    
    print("\nMeta-learning systems testing completed successfully!")