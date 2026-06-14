"""
Transfer Learning for Financial Models
Advanced techniques for transferring knowledge across financial domains and tasks
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
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class TransferLearner(ABC):
    """Abstract base class for transfer learning methods"""
    
    @abstractmethod
    def pretrain(
        self,
        source_data: Tuple[torch.Tensor, torch.Tensor],
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """
        Pre-train on source domain
        
        Args:
            source_data: (X_source, y_source) data from source domain
            n_epochs: Number of pre-training epochs
        
        Returns:
            Pre-training metrics
        """
        pass
    
    @abstractmethod
    def transfer(
        self,
        target_data: Tuple[torch.Tensor, torch.Tensor],
        strategy: str = 'fine_tune',
        n_epochs: int = 50
    ) -> Dict[str, float]:
        """
        Transfer to target domain
        
        Args:
            target_data: (X_target, y_target) data from target domain
            strategy: Transfer strategy ('fine_tune', 'feature_extraction', 'domain_adaptation')
            n_epochs: Number of transfer epochs
        
        Returns:
            Transfer learning metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        pass


class FineTuningTransfer(TransferLearner):
    """
    Fine-tuning based transfer learning
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        freeze_layers: Optional[List[str]] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.freeze_layers = freeze_layers or []
        
        self.source_optimizer = None
        self.target_optimizer = None
        self.pretrained = False
        
    def _freeze_layers(self):
        """Freeze specified layers"""
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in self.freeze_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def pretrain(
        self,
        source_data: Tuple[torch.Tensor, torch.Tensor],
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """Pre-train on source domain"""
        
        X_source, y_source = source_data
        
        # Enable all parameters for pre-training
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.source_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        losses = []
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_source, y_source)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.source_optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                
                # Determine loss function based on target type
                if y_source.dtype == torch.long:
                    loss = F.cross_entropy(outputs, batch_y)
                else:
                    loss = F.mse_loss(outputs.squeeze(), batch_y)
                
                loss.backward()
                self.source_optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(dataloader))
            
            if epoch % 20 == 0:
                print(f"Pre-training Epoch {epoch}, Loss: {losses[-1]:.4f}")
        
        self.pretrained = True
        return {'pretrain_losses': losses}
    
    def transfer(
        self,
        target_data: Tuple[torch.Tensor, torch.Tensor],
        strategy: str = 'fine_tune',
        n_epochs: int = 50
    ) -> Dict[str, float]:
        """Transfer to target domain"""
        
        if not self.pretrained:
            raise ValueError("Model must be pre-trained before transfer")
        
        X_target, y_target = target_data
        
        if strategy == 'fine_tune':
            # Fine-tune all or selected layers
            self._freeze_layers()
            
            # Create optimizer for unfrozen parameters only
            unfrozen_params = [p for p in self.model.parameters() if p.requires_grad]
            self.target_optimizer = optim.Adam(unfrozen_params, lr=self.learning_rate * 0.1)
            
        elif strategy == 'feature_extraction':
            # Freeze all layers except the last one
            for name, param in self.model.named_parameters():
                if 'classifier' not in name and 'fc' not in name and 'output' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            
            unfrozen_params = [p for p in self.model.parameters() if p.requires_grad]
            self.target_optimizer = optim.Adam(unfrozen_params, lr=self.learning_rate)
        
        else:
            raise ValueError(f"Unknown transfer strategy: {strategy}")
        
        # Fine-tuning loop
        self.model.train()
        losses = []
        accuracies = []
        
        dataset = torch.utils.data.TensorDataset(X_target, y_target)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                self.target_optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                
                if y_target.dtype == torch.long:
                    loss = F.cross_entropy(outputs, batch_y)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                else:
                    loss = F.mse_loss(outputs.squeeze(), batch_y)
                
                loss.backward()
                self.target_optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(dataloader))
            
            if y_target.dtype == torch.long:
                accuracies.append(correct / total)
            
            if epoch % 10 == 0:
                if accuracies:
                    print(f"Transfer Epoch {epoch}, Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}")
                else:
                    print(f"Transfer Epoch {epoch}, Loss: {losses[-1]:.4f}")
        
        results = {'transfer_losses': losses}
        if accuracies:
            results['transfer_accuracies'] = accuracies
        
        return results
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            return self.model(X)


class DomainAdaptationTransfer(TransferLearner):
    """
    Domain Adaptation with adversarial training
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        domain_discriminator: nn.Module,
        learning_rate: float = 0.001,
        lambda_domain: float = 0.1
    ):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_discriminator = domain_discriminator
        self.learning_rate = learning_rate
        self.lambda_domain = lambda_domain
        
        self.fe_optimizer = None
        self.cls_optimizer = None
        self.disc_optimizer = None
        self.pretrained = False
        
    def pretrain(
        self,
        source_data: Tuple[torch.Tensor, torch.Tensor],
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """Pre-train on source domain"""
        
        X_source, y_source = source_data
        
        # Optimizers
        self.fe_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=self.learning_rate)
        self.cls_optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        
        self.feature_extractor.train()
        self.classifier.train()
        
        losses = []
        
        dataset = torch.utils.data.TensorDataset(X_source, y_source)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Extract features
                features = self.feature_extractor(batch_X)
                
                # Classify
                outputs = self.classifier(features)
                
                # Classification loss
                if y_source.dtype == torch.long:
                    loss = F.cross_entropy(outputs, batch_y)
                else:
                    loss = F.mse_loss(outputs.squeeze(), batch_y)
                
                # Update feature extractor and classifier
                self.fe_optimizer.zero_grad()
                self.cls_optimizer.zero_grad()
                loss.backward()
                self.fe_optimizer.step()
                self.cls_optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(dataloader))
            
            if epoch % 20 == 0:
                print(f"Pre-training Epoch {epoch}, Loss: {losses[-1]:.4f}")
        
        self.pretrained = True
        return {'pretrain_losses': losses}
    
    def transfer(
        self,
        target_data: Tuple[torch.Tensor, torch.Tensor],
        strategy: str = 'domain_adaptation',
        n_epochs: int = 50
    ) -> Dict[str, float]:
        """Transfer using domain adaptation"""
        
        if not self.pretrained:
            raise ValueError("Model must be pre-trained before transfer")
        
        X_target, y_target = target_data
        
        # Create combined dataset
        source_size = 1000  # Assume we keep some source data
        X_source_subset = torch.randn(source_size, X_target.size(1))  # Placeholder
        
        # Domain labels (0 for source, 1 for target)
        domain_labels_source = torch.zeros(source_size, dtype=torch.long)
        domain_labels_target = torch.ones(X_target.size(0), dtype=torch.long)
        
        # Discriminator optimizer
        self.disc_optimizer = optim.Adam(self.domain_discriminator.parameters(), lr=self.learning_rate)
        
        losses = []
        cls_losses = []
        domain_losses = []
        
        for epoch in range(n_epochs):
            epoch_cls_loss = 0.0
            epoch_domain_loss = 0.0
            
            # Create mini-batches
            batch_size = 32
            n_batches = max(len(X_target) // batch_size, 1)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X_target))
                
                batch_X_target = X_target[start_idx:end_idx]
                batch_y_target = y_target[start_idx:end_idx]
                
                # Sample source data
                source_indices = torch.randint(0, source_size, (len(batch_X_target),))
                batch_X_source = X_source_subset[source_indices]
                
                batch_domain_source = domain_labels_source[source_indices]
                batch_domain_target = domain_labels_target[start_idx:end_idx]
                
                # Train domain discriminator
                self.disc_optimizer.zero_grad()
                
                # Features from both domains
                features_source = self.feature_extractor(batch_X_source).detach()
                features_target = self.feature_extractor(batch_X_target).detach()
                
                # Domain predictions
                domain_pred_source = self.domain_discriminator(features_source)
                domain_pred_target = self.domain_discriminator(features_target)
                
                # Domain loss
                domain_loss = (F.cross_entropy(domain_pred_source, batch_domain_source) +
                              F.cross_entropy(domain_pred_target, batch_domain_target))
                
                domain_loss.backward()
                self.disc_optimizer.step()
                
                # Train feature extractor and classifier
                self.fe_optimizer.zero_grad()
                self.cls_optimizer.zero_grad()
                
                # Features (with gradients)
                features_target = self.feature_extractor(batch_X_target)
                
                # Classification loss
                cls_outputs = self.classifier(features_target)
                if y_target.dtype == torch.long:
                    cls_loss = F.cross_entropy(cls_outputs, batch_y_target)
                else:
                    cls_loss = F.mse_loss(cls_outputs.squeeze(), batch_y_target)
                
                # Adversarial domain loss (fool the discriminator)
                domain_pred = self.domain_discriminator(features_target)
                # Try to make target look like source (label 0)
                adversarial_loss = F.cross_entropy(domain_pred, torch.zeros_like(batch_domain_target))
                
                # Combined loss
                total_loss = cls_loss + self.lambda_domain * adversarial_loss
                
                total_loss.backward()
                self.fe_optimizer.step()
                self.cls_optimizer.step()
                
                epoch_cls_loss += cls_loss.item()
                epoch_domain_loss += domain_loss.item()
            
            cls_losses.append(epoch_cls_loss / n_batches)
            domain_losses.append(epoch_domain_loss / n_batches)
            losses.append(cls_losses[-1] + self.lambda_domain * domain_losses[-1])
            
            if epoch % 10 == 0:
                print(f"Transfer Epoch {epoch}, Cls Loss: {cls_losses[-1]:.4f}, Domain Loss: {domain_losses[-1]:.4f}")
        
        return {
            'transfer_losses': losses,
            'classification_losses': cls_losses,
            'domain_losses': domain_losses
        }
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        self.feature_extractor.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            features = self.feature_extractor(X)
            return self.classifier(features)


class ProgressiveTransfer(TransferLearner):
    """
    Progressive transfer learning with gradual unfreezing
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        unfreeze_schedule: Optional[Dict[int, List[str]]] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.unfreeze_schedule = unfreeze_schedule or {}
        
        self.optimizer = None
        self.pretrained = False
        self.frozen_layers = set()
        
    def _freeze_all_except(self, layer_names: List[str]):
        """Freeze all layers except specified ones"""
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                self.frozen_layers.discard(name)
            else:
                param.requires_grad = False
                self.frozen_layers.add(name)
    
    def pretrain(
        self,
        source_data: Tuple[torch.Tensor, torch.Tensor],
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """Pre-train on source domain"""
        
        X_source, y_source = source_data
        
        # Enable all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        losses = []
        
        dataset = torch.utils.data.TensorDataset(X_source, y_source)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                
                if y_source.dtype == torch.long:
                    loss = F.cross_entropy(outputs, batch_y)
                else:
                    loss = F.mse_loss(outputs.squeeze(), batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(dataloader))
            
            if epoch % 20 == 0:
                print(f"Pre-training Epoch {epoch}, Loss: {losses[-1]:.4f}")
        
        self.pretrained = True
        return {'pretrain_losses': losses}
    
    def transfer(
        self,
        target_data: Tuple[torch.Tensor, torch.Tensor],
        strategy: str = 'progressive',
        n_epochs: int = 50
    ) -> Dict[str, float]:
        """Progressive transfer learning"""
        
        if not self.pretrained:
            raise ValueError("Model must be pre-trained before transfer")
        
        X_target, y_target = target_data
        
        # Start with only the last layer unfrozen
        layer_names = [name.split('.')[0] for name, _ in self.model.named_parameters()]
        unique_layers = list(dict.fromkeys(layer_names))  # Preserve order
        
        # Initially freeze all except last layer
        self._freeze_all_except([unique_layers[-1]])
        
        self.model.train()
        losses = []
        unfrozen_history = []
        
        dataset = torch.utils.data.TensorDataset(X_target, y_target)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(n_epochs):
            # Check unfreeze schedule
            if epoch in self.unfreeze_schedule:
                layers_to_unfreeze = self.unfreeze_schedule[epoch]
                current_unfrozen = [name.split('.')[0] for name, param in self.model.named_parameters() 
                                  if param.requires_grad]
                new_unfrozen = list(set(current_unfrozen + layers_to_unfreeze))
                self._freeze_all_except(new_unfrozen)
                print(f"Epoch {epoch}: Unfrozen layers: {new_unfrozen}")
            
            # Or automatic progressive unfreezing
            elif strategy == 'progressive' and epoch > 0 and epoch % 10 == 0:
                # Unfreeze one more layer from the end
                current_unfrozen = [name.split('.')[0] for name, param in self.model.named_parameters() 
                                  if param.requires_grad]
                
                if len(current_unfrozen) < len(unique_layers):
                    # Find next layer to unfreeze (working backwards)
                    for layer_name in reversed(unique_layers):
                        if layer_name not in current_unfrozen:
                            current_unfrozen.append(layer_name)
                            break
                    
                    self._freeze_all_except(current_unfrozen)
                    print(f"Epoch {epoch}: Progressively unfrozen layers: {current_unfrozen}")
            
            # Update optimizer with current parameters
            unfrozen_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(unfrozen_params, lr=self.learning_rate * 0.1)
            
            # Track unfrozen layers
            unfrozen_layers = [name.split('.')[0] for name, param in self.model.named_parameters() 
                             if param.requires_grad]
            unfrozen_history.append(len(set(unfrozen_layers)))
            
            # Training loop
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                
                if y_target.dtype == torch.long:
                    loss = F.cross_entropy(outputs, batch_y)
                else:
                    loss = F.mse_loss(outputs.squeeze(), batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(dataloader))
            
            if epoch % 10 == 0:
                print(f"Transfer Epoch {epoch}, Loss: {losses[-1]:.4f}, Unfrozen Layers: {unfrozen_history[-1]}")
        
        return {
            'transfer_losses': losses,
            'unfrozen_layers_history': unfrozen_history
        }
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            return self.model(X)


class FinancialTransferLearner:
    """
    Financial-specific transfer learning coordinator
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        transfer_method: str = 'fine_tuning',
        market_adaptations: Optional[Dict[str, Any]] = None
    ):
        self.base_model = base_model
        self.transfer_method = transfer_method
        self.market_adaptations = market_adaptations or {}
        
        self.transfer_learner = None
        self.scalers = {}
        self.transfer_history = []
        
        self._initialize_transfer_learner()
    
    def _initialize_transfer_learner(self):
        """Initialize appropriate transfer learner"""
        
        if self.transfer_method == 'fine_tuning':
            self.transfer_learner = FineTuningTransfer(
                model=self.base_model,
                freeze_layers=self.market_adaptations.get('freeze_layers', [])
            )
        
        elif self.transfer_method == 'progressive':
            self.transfer_learner = ProgressiveTransfer(
                model=self.base_model,
                unfreeze_schedule=self.market_adaptations.get('unfreeze_schedule', {})
            )
        
        elif self.transfer_method == 'domain_adaptation':
            # Extract components for domain adaptation
            if hasattr(self.base_model, 'feature_extractor'):
                feature_extractor = self.base_model.feature_extractor
                classifier = self.base_model.classifier
            else:
                # Split model into feature extractor and classifier
                layers = list(self.base_model.children())
                feature_extractor = nn.Sequential(*layers[:-1])
                classifier = layers[-1]
            
            # Create domain discriminator
            feature_dim = self.market_adaptations.get('feature_dim', 128)
            domain_discriminator = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # Binary domain classification
            )
            
            self.transfer_learner = DomainAdaptationTransfer(
                feature_extractor=feature_extractor,
                classifier=classifier,
                domain_discriminator=domain_discriminator
            )
        
        else:
            raise ValueError(f"Unknown transfer method: {self.transfer_method}")
    
    def pretrain_on_market(
        self,
        market_data: Dict[str, pd.DataFrame],
        market_name: str,
        target_column: str = 'return',
        window_size: int = 20,
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """
        Pre-train on source market data
        
        Args:
            market_data: Dictionary of market DataFrames
            market_name: Name of source market
            target_column: Target variable column name
            window_size: Size of input windows
            n_epochs: Pre-training epochs
        
        Returns:
            Pre-training results
        """
        
        if market_name not in market_data:
            raise ValueError(f"Market {market_name} not found in data")
        
        # Prepare data
        X_source, y_source = self._prepare_financial_data(
            market_data[market_name], target_column, window_size
        )
        
        # Scale data
        scaler_X = StandardScaler()
        X_source_scaled = torch.FloatTensor(scaler_X.fit_transform(X_source.numpy()))
        self.scalers[f'{market_name}_X'] = scaler_X
        
        if y_source.dtype != torch.long:
            scaler_y = StandardScaler()
            y_source_scaled = torch.FloatTensor(scaler_y.fit_transform(y_source.numpy().reshape(-1, 1)).flatten())
            self.scalers[f'{market_name}_y'] = scaler_y
        else:
            y_source_scaled = y_source
        
        # Pre-train
        results = self.transfer_learner.pretrain((X_source_scaled, y_source_scaled), n_epochs)
        
        print(f"Pre-training completed on {market_name}")
        return results
    
    def transfer_to_market(
        self,
        market_data: Dict[str, pd.DataFrame],
        target_market: str,
        target_column: str = 'return',
        window_size: int = 20,
        n_epochs: int = 50,
        strategy: str = 'fine_tune'
    ) -> Dict[str, float]:
        """
        Transfer to target market
        
        Args:
            market_data: Dictionary of market DataFrames
            target_market: Name of target market
            target_column: Target variable column name
            window_size: Size of input windows
            n_epochs: Transfer epochs
            strategy: Transfer strategy
        
        Returns:
            Transfer results
        """
        
        if target_market not in market_data:
            raise ValueError(f"Market {target_market} not found in data")
        
        # Prepare target data
        X_target, y_target = self._prepare_financial_data(
            market_data[target_market], target_column, window_size
        )
        
        # Scale data (fit new scalers for target domain)
        scaler_X = StandardScaler()
        X_target_scaled = torch.FloatTensor(scaler_X.fit_transform(X_target.numpy()))
        self.scalers[f'{target_market}_X'] = scaler_X
        
        if y_target.dtype != torch.long:
            scaler_y = StandardScaler()
            y_target_scaled = torch.FloatTensor(scaler_y.fit_transform(y_target.numpy().reshape(-1, 1)).flatten())
            self.scalers[f'{target_market}_y'] = scaler_y
        else:
            y_target_scaled = y_target
        
        # Transfer
        results = self.transfer_learner.transfer((X_target_scaled, y_target_scaled), strategy, n_epochs)
        
        # Record transfer
        self.transfer_history.append({
            'target_market': target_market,
            'strategy': strategy,
            'n_epochs': n_epochs,
            'results': results
        })
        
        print(f"Transfer completed to {target_market}")
        return results
    
    def _prepare_financial_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        window_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare financial data for training"""
        
        # Create features and targets
        features = []
        targets = []
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                data[col] = data.get('price', data.iloc[:, 0])  # Fallback
        
        # Calculate additional features
        data = data.copy()
        data['return'] = data['close'].pct_change()
        data['volatility'] = data['return'].rolling(window=10).std()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        
        # Create windows
        for i in range(window_size, len(data)):
            # Feature window
            window_data = data.iloc[i-window_size:i][required_cols + ['return', 'volatility']].values
            features.append(window_data.flatten())
            
            # Target
            if target_column == 'direction':
                # Binary classification: up/down
                target = 1 if data.iloc[i]['close'] > data.iloc[i-1]['close'] else 0
                targets.append(target)
            else:
                # Regression: return or other continuous target
                target = data.iloc[i][target_column] if target_column in data.columns else data.iloc[i]['return']
                targets.append(target if not np.isnan(target) else 0.0)
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(features))
        
        if target_column == 'direction':
            y = torch.LongTensor(targets)
        else:
            y = torch.FloatTensor(targets)
        
        return X, y
    
    def predict_market(
        self,
        market_data: pd.DataFrame,
        market_name: str,
        window_size: int = 20
    ) -> np.ndarray:
        """Make predictions on market data"""
        
        # Prepare data
        X, _ = self._prepare_financial_data(market_data, 'return', window_size)
        
        # Scale using appropriate scaler
        if f'{market_name}_X' in self.scalers:
            X_scaled = torch.FloatTensor(self.scalers[f'{market_name}_X'].transform(X.numpy()))
        else:
            # Use the last available scaler
            scaler_key = list(self.scalers.keys())[-1] if '_X' in str(self.scalers.keys()) else None
            if scaler_key and '_X' in scaler_key:
                X_scaled = torch.FloatTensor(self.scalers[scaler_key].transform(X.numpy()))
            else:
                X_scaled = X
        
        # Predict
        predictions = self.transfer_learner.predict(X_scaled)
        
        # Inverse transform if needed
        if predictions.dim() == 1 and f'{market_name}_y' in self.scalers:
            predictions_np = self.scalers[f'{market_name}_y'].inverse_transform(
                predictions.numpy().reshape(-1, 1)
            ).flatten()
        else:
            predictions_np = predictions.numpy()
        
        return predictions_np
    
    def evaluate_transfer_performance(
        self,
        market_data: Dict[str, pd.DataFrame],
        test_markets: List[str],
        target_column: str = 'return',
        window_size: int = 20
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate transfer performance across markets"""
        
        results = {}
        
        for market in test_markets:
            if market not in market_data:
                continue
            
            # Prepare test data
            X_test, y_test = self._prepare_financial_data(
                market_data[market], target_column, window_size
            )
            
            # Make predictions
            predictions = self.predict_market(market_data[market], market, window_size)
            
            # Evaluate
            if y_test.dtype == torch.long:
                # Classification metrics
                accuracy = accuracy_score(y_test.numpy(), predictions.round())
                results[market] = {'accuracy': accuracy}
            else:
                # Regression metrics
                mse = mean_squared_error(y_test.numpy(), predictions)
                r2 = r2_score(y_test.numpy(), predictions)
                results[market] = {'mse': mse, 'r2': r2}
        
        return results


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Testing Transfer Learning Methods...")
    
    # Create test model
    class TestMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU()
            )
            self.classifier = nn.Linear(hidden_dim//2, output_dim)
            
        def forward(self, x):
            features = self.feature_extractor(x)
            return self.classifier(features)
    
    # Test with synthetic data
    input_dim = 50
    hidden_dim = 128
    output_dim = 2
    
    model = TestMLP(input_dim, hidden_dim, output_dim)
    
    # Generate synthetic source domain data
    X_source = torch.randn(1000, input_dim)
    y_source = torch.randint(0, output_dim, (1000,))
    
    # Generate synthetic target domain data (slightly different distribution)
    X_target = torch.randn(500, input_dim) + 0.5  # Domain shift
    y_target = torch.randint(0, output_dim, (500,))
    
    # Test Fine-tuning Transfer
    print("\nTesting Fine-tuning Transfer...")
    try:
        ft_transfer = FineTuningTransfer(copy.deepcopy(model))
        
        # Pre-train
        ft_results = ft_transfer.pretrain((X_source, y_source), n_epochs=50)
        print("Fine-tuning pre-training completed")
        
        # Transfer
        transfer_results = ft_transfer.transfer((X_target, y_target), strategy='fine_tune', n_epochs=30)
        print("Fine-tuning transfer completed")
        
        # Test prediction
        predictions = ft_transfer.predict(X_target[:10])
        print(f"Sample predictions shape: {predictions.shape}")
        
    except Exception as e:
        print(f"Fine-tuning transfer error: {e}")
    
    # Test Progressive Transfer
    print("\nTesting Progressive Transfer...")
    try:
        prog_transfer = ProgressiveTransfer(copy.deepcopy(model))
        
        # Pre-train
        prog_transfer.pretrain((X_source, y_source), n_epochs=50)
        print("Progressive pre-training completed")
        
        # Transfer with progressive unfreezing
        prog_results = prog_transfer.transfer((X_target, y_target), strategy='progressive', n_epochs=40)
        print("Progressive transfer completed")
        
    except Exception as e:
        print(f"Progressive transfer error: {e}")
    
    # Test Financial Transfer Learning
    print("\nTesting Financial Transfer Learning...")
    try:
        # Generate synthetic financial data
        markets = ['US_STOCKS', 'EU_STOCKS', 'ASIA_STOCKS']
        market_data = {}
        
        for market in markets:
            n_days = 1000
            base_price = 100
            
            # Different market characteristics
            if market == 'US_STOCKS':
                returns = np.random.normal(0.0008, 0.02, n_days)  # Higher growth
            elif market == 'EU_STOCKS':
                returns = np.random.normal(0.0003, 0.015, n_days)  # Moderate growth
            else:  # ASIA_STOCKS
                returns = np.random.normal(0.0005, 0.025, n_days)  # Higher volatility
            
            prices = base_price * np.cumprod(1 + returns)
            
            # Create OHLCV data
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_days))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_days))),
                'close': prices,
                'volume': np.random.exponential(1000000, n_days)
            })
            
            market_data[market] = df
        
        # Create financial transfer learner
        financial_model = TestMLP(input_dim=140, hidden_dim=256, output_dim=1)  # 20 windows * 7 features
        
        financial_transfer = FinancialTransferLearner(
            base_model=financial_model,
            transfer_method='fine_tuning'
        )
        
        # Pre-train on US market
        pretrain_results = financial_transfer.pretrain_on_market(
            market_data, 'US_STOCKS', target_column='return', n_epochs=30
        )
        
        # Transfer to EU market
        transfer_results = financial_transfer.transfer_to_market(
            market_data, 'EU_STOCKS', target_column='return', n_epochs=20
        )
        
        # Evaluate on all markets
        evaluation = financial_transfer.evaluate_transfer_performance(
            market_data, ['US_STOCKS', 'EU_STOCKS', 'ASIA_STOCKS']
        )
        
        print("Financial Transfer Learning Results:")
        for market, metrics in evaluation.items():
            print(f"  {market}: {metrics}")
        
    except Exception as e:
        print(f"Financial transfer learning error: {e}")
    
    print("\nDone!")