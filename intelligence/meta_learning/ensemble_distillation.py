"""
Ensemble Distillation for Financial Models
Knowledge distillation techniques for compressing ensemble models
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
import matplotlib.pyplot as plt


class EnsembleDistiller(ABC):
    """Abstract base class for ensemble distillation methods"""
    
    @abstractmethod
    def distill(
        self,
        teacher_ensemble: List[nn.Module],
        student_model: nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        temperature: float = 3.0,
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """
        Distill knowledge from teacher ensemble to student model
        
        Args:
            teacher_ensemble: List of teacher models
            student_model: Student model to train
            train_data: Training data (X, y)
            temperature: Temperature for softening predictions
            n_epochs: Number of training epochs
        
        Returns:
            Training metrics
        """
        pass
    
    @abstractmethod
    def evaluate_compression(
        self,
        teacher_ensemble: List[nn.Module],
        student_model: nn.Module,
        test_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate compression performance
        
        Args:
            teacher_ensemble: Teacher models
            student_model: Student model
            test_data: Test data
        
        Returns:
            Evaluation metrics
        """
        pass


class StandardDistillation(EnsembleDistiller):
    """
    Standard knowledge distillation with temperature scaling
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        alpha: float = 0.5,  # Weight for distillation loss
        beta: float = 0.5    # Weight for student loss
    ):
        self.learning_rate = learning_rate
        self.alpha = alpha  # Distillation loss weight
        self.beta = beta    # Student loss weight
    
    def _ensemble_predictions(
        self,
        teacher_ensemble: List[nn.Module],
        X: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Get ensemble predictions from teacher models"""
        
        predictions = []
        
        for teacher in teacher_ensemble:
            teacher.eval()
            with torch.no_grad():
                pred = teacher(X)
                if temperature != 1.0:
                    pred = pred / temperature
                predictions.append(F.softmax(pred, dim=1) if pred.dim() > 1 else pred)
        
        # Average ensemble predictions
        if len(predictions[0].shape) > 1:
            # Classification: average probabilities
            ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        else:
            # Regression: average outputs
            ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        return ensemble_pred
    
    def distill(
        self,
        teacher_ensemble: List[nn.Module],
        student_model: nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        temperature: float = 3.0,
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """Standard knowledge distillation"""
        
        X_train, y_train = train_data
        
        optimizer = optim.Adam(student_model.parameters(), lr=self.learning_rate)
        
        student_model.train()
        losses = []
        distill_losses = []
        student_losses = []
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_distill_loss = 0.0
            epoch_student_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Student predictions
                student_outputs = student_model(batch_X)
                
                # Teacher ensemble predictions (soft targets)
                teacher_soft_targets = self._ensemble_predictions(
                    teacher_ensemble, batch_X, temperature
                )
                
                # Distillation loss (KL divergence between student and teacher)
                if teacher_soft_targets.dim() > 1:
                    # Classification
                    student_soft = F.log_softmax(student_outputs / temperature, dim=1)
                    distill_loss = F.kl_div(student_soft, teacher_soft_targets, reduction='batchmean')
                else:
                    # Regression
                    distill_loss = F.mse_loss(student_outputs.squeeze(), teacher_soft_targets)
                
                # Student loss (original task loss)
                if y_train.dtype == torch.long:
                    student_loss = F.cross_entropy(student_outputs, batch_y)
                else:
                    student_loss = F.mse_loss(student_outputs.squeeze(), batch_y)
                
                # Combined loss
                total_loss = self.alpha * distill_loss + self.beta * student_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_distill_loss += distill_loss.item()
                epoch_student_loss += student_loss.item()
            
            # Average losses
            losses.append(epoch_loss / len(dataloader))
            distill_losses.append(epoch_distill_loss / len(dataloader))
            student_losses.append(epoch_student_loss / len(dataloader))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Total Loss: {losses[-1]:.4f}, "
                      f"Distill: {distill_losses[-1]:.4f}, Student: {student_losses[-1]:.4f}")
        
        return {
            'total_losses': losses,
            'distillation_losses': distill_losses,
            'student_losses': student_losses
        }
    
    def evaluate_compression(
        self,
        teacher_ensemble: List[nn.Module],
        student_model: nn.Module,
        test_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate compression performance"""
        
        X_test, y_test = test_data
        
        # Teacher ensemble predictions
        teacher_predictions = self._ensemble_predictions(teacher_ensemble, X_test)
        
        # Student predictions
        student_model.eval()
        with torch.no_grad():
            student_predictions = student_model(X_test)
        
        results = {}
        
        if y_test.dtype == torch.long:
            # Classification metrics
            teacher_acc = accuracy_score(
                y_test.numpy(),
                teacher_predictions.argmax(dim=1).numpy()
            )
            student_acc = accuracy_score(
                y_test.numpy(),
                student_predictions.argmax(dim=1).numpy()
            )
            
            results.update({
                'teacher_accuracy': teacher_acc,
                'student_accuracy': student_acc,
                'accuracy_retention': student_acc / teacher_acc if teacher_acc > 0 else 0
            })
        
        else:
            # Regression metrics
            teacher_mse = mean_squared_error(y_test.numpy(), teacher_predictions.numpy())
            student_mse = mean_squared_error(y_test.numpy(), student_predictions.squeeze().numpy())
            
            teacher_r2 = r2_score(y_test.numpy(), teacher_predictions.numpy())
            student_r2 = r2_score(y_test.numpy(), student_predictions.squeeze().numpy())
            
            results.update({
                'teacher_mse': teacher_mse,
                'student_mse': student_mse,
                'teacher_r2': teacher_r2,
                'student_r2': student_r2,
                'r2_retention': student_r2 / teacher_r2 if teacher_r2 > 0 else 0
            })
        
        # Model size comparison
        teacher_params = sum(sum(p.numel() for p in model.parameters()) for model in teacher_ensemble)
        student_params = sum(p.numel() for p in student_model.parameters())
        
        results.update({
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'compression_ratio': teacher_params / student_params if student_params > 0 else 0
        })
        
        return results


class AttentionDistillation(EnsembleDistiller):
    """
    Attention-based knowledge distillation
    Focuses on important features learned by teachers
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        attention_loss_weight: float = 0.3,
        feature_loss_weight: float = 0.3,
        task_loss_weight: float = 0.4
    ):
        self.learning_rate = learning_rate
        self.attention_loss_weight = attention_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.task_loss_weight = task_loss_weight
    
    def _extract_attention_maps(
        self,
        model: nn.Module,
        X: torch.Tensor
    ) -> List[torch.Tensor]:
        """Extract attention maps from intermediate layers"""
        
        attention_maps = []
        
        def hook_fn(module, input, output):
            # Compute attention as channel-wise mean
            if len(output.shape) >= 2:
                attention = torch.mean(output, dim=1, keepdim=True)
                attention_maps.append(attention)
        
        # Register hooks on intermediate layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        model(X)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def _attention_loss(
        self,
        teacher_attentions: List[List[torch.Tensor]],
        student_attentions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention transfer loss"""
        
        if not teacher_attentions or not student_attentions:
            return torch.tensor(0.0)
        
        # Average teacher attentions
        n_teachers = len(teacher_attentions)
        n_layers = min(len(att) for att in teacher_attentions)
        
        total_loss = 0.0
        count = 0
        
        for layer_idx in range(min(n_layers, len(student_attentions))):
            # Average teacher attention for this layer
            teacher_att_avg = torch.zeros_like(student_attentions[layer_idx])
            
            for teacher_att in teacher_attentions:
                if layer_idx < len(teacher_att):
                    # Resize if necessary
                    if teacher_att[layer_idx].shape != student_attentions[layer_idx].shape:
                        teacher_resized = F.adaptive_avg_pool1d(
                            teacher_att[layer_idx].view(teacher_att[layer_idx].size(0), -1).unsqueeze(1),
                            student_attentions[layer_idx].view(student_attentions[layer_idx].size(0), -1).size(1)
                        ).squeeze(1).view_as(student_attentions[layer_idx])
                    else:
                        teacher_resized = teacher_att[layer_idx]
                    
                    teacher_att_avg += teacher_resized / n_teachers
            
            # Normalize attention maps
            teacher_att_norm = F.normalize(teacher_att_avg.view(teacher_att_avg.size(0), -1), p=2, dim=1)
            student_att_norm = F.normalize(student_attentions[layer_idx].view(student_attentions[layer_idx].size(0), -1), p=2, dim=1)
            
            # Compute similarity loss
            layer_loss = F.mse_loss(student_att_norm, teacher_att_norm)
            total_loss += layer_loss
            count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0)
    
    def distill(
        self,
        teacher_ensemble: List[nn.Module],
        student_model: nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        temperature: float = 3.0,
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """Attention-based distillation"""
        
        X_train, y_train = train_data
        
        optimizer = optim.Adam(student_model.parameters(), lr=self.learning_rate)
        
        student_model.train()
        for teacher in teacher_ensemble:
            teacher.eval()
        
        losses = []
        attention_losses = []
        task_losses = []
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_attention_loss = 0.0
            epoch_task_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Extract teacher attention maps
                teacher_attentions = []
                teacher_predictions = []
                
                for teacher in teacher_ensemble:
                    teacher.eval()
                    with torch.no_grad():
                        teacher_att = self._extract_attention_maps(teacher, batch_X)
                        teacher_pred = teacher(batch_X)
                        
                        teacher_attentions.append(teacher_att)
                        teacher_predictions.append(teacher_pred)
                
                # Student forward pass and attention extraction
                student_attentions = self._extract_attention_maps(student_model, batch_X)
                student_predictions = student_model(batch_X)
                
                # Task loss
                if y_train.dtype == torch.long:
                    task_loss = F.cross_entropy(student_predictions, batch_y)
                else:
                    task_loss = F.mse_loss(student_predictions.squeeze(), batch_y)
                
                # Attention transfer loss
                attention_loss = self._attention_loss(teacher_attentions, student_attentions)
                
                # Feature distillation loss (ensemble predictions)
                ensemble_pred = torch.stack(teacher_predictions, dim=0).mean(dim=0)
                if ensemble_pred.dim() > 1:
                    feature_loss = F.kl_div(
                        F.log_softmax(student_predictions / temperature, dim=1),
                        F.softmax(ensemble_pred / temperature, dim=1),
                        reduction='batchmean'
                    )
                else:
                    feature_loss = F.mse_loss(student_predictions.squeeze(), ensemble_pred)
                
                # Combined loss
                total_loss = (self.task_loss_weight * task_loss +
                             self.attention_loss_weight * attention_loss +
                             self.feature_loss_weight * feature_loss)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_attention_loss += attention_loss.item()
                epoch_task_loss += task_loss.item()
            
            losses.append(epoch_loss / len(dataloader))
            attention_losses.append(epoch_attention_loss / len(dataloader))
            task_losses.append(epoch_task_loss / len(dataloader))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Total Loss: {losses[-1]:.4f}, "
                      f"Attention: {attention_losses[-1]:.4f}, Task: {task_losses[-1]:.4f}")
        
        return {
            'total_losses': losses,
            'attention_losses': attention_losses,
            'task_losses': task_losses
        }
    
    def evaluate_compression(
        self,
        teacher_ensemble: List[nn.Module],
        student_model: nn.Module,
        test_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate compression with attention analysis"""
        
        # Use standard evaluation
        standard_distiller = StandardDistillation()
        results = standard_distiller.evaluate_compression(teacher_ensemble, student_model, test_data)
        
        # Add attention similarity metric
        X_test, _ = test_data
        
        # Extract attention maps
        teacher_attentions = []
        for teacher in teacher_ensemble:
            teacher.eval()
            teacher_att = self._extract_attention_maps(teacher, X_test[:100])  # Sample for efficiency
            teacher_attentions.append(teacher_att)
        
        student_model.eval()
        student_attentions = self._extract_attention_maps(student_model, X_test[:100])
        
        # Compute attention similarity
        attention_similarity = 1.0 - self._attention_loss(teacher_attentions, student_attentions).item()
        results['attention_similarity'] = max(0, attention_similarity)
        
        return results


class ProgressiveDistillation(EnsembleDistiller):
    """
    Progressive distillation with curriculum learning
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        n_stages: int = 3,
        stage_epochs: int = 30
    ):
        self.learning_rate = learning_rate
        self.n_stages = n_stages
        self.stage_epochs = stage_epochs
    
    def _create_curriculum(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        teacher_ensemble: List[nn.Module]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create curriculum based on teacher disagreement"""
        
        # Get ensemble predictions
        predictions = []
        for teacher in teacher_ensemble:
            teacher.eval()
            with torch.no_grad():
                pred = teacher(X_train)
                predictions.append(pred)
        
        # Calculate disagreement (variance across teachers)
        pred_stack = torch.stack(predictions, dim=0)
        if pred_stack.dim() > 2:
            # Classification: use prediction variance
            pred_probs = F.softmax(pred_stack, dim=-1)
            disagreement = torch.var(pred_probs, dim=0).mean(dim=1)
        else:
            # Regression: use output variance
            disagreement = torch.var(pred_stack, dim=0)
        
        # Sort by disagreement (easy to hard)
        sorted_indices = torch.argsort(disagreement)
        
        # Create stages
        n_samples = len(X_train)
        stage_size = n_samples // self.n_stages
        
        stages = []
        for stage in range(self.n_stages):
            start_idx = stage * stage_size
            end_idx = n_samples if stage == self.n_stages - 1 else (stage + 1) * stage_size
            
            stage_indices = sorted_indices[start_idx:end_idx]
            stages.append((X_train[stage_indices], y_train[stage_indices]))
        
        return stages
    
    def distill(
        self,
        teacher_ensemble: List[nn.Module],
        student_model: nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        temperature: float = 3.0,
        n_epochs: int = 100
    ) -> Dict[str, float]:
        """Progressive distillation with curriculum"""
        
        X_train, y_train = train_data
        
        # Create curriculum
        curriculum_stages = self._create_curriculum(X_train, y_train, teacher_ensemble)
        
        optimizer = optim.Adam(student_model.parameters(), lr=self.learning_rate)
        
        all_losses = []
        stage_performances = []
        
        # Train on each stage progressively
        for stage_idx, (stage_X, stage_y) in enumerate(curriculum_stages):
            print(f"Training on stage {stage_idx + 1}/{self.n_stages} ({len(stage_X)} samples)")
            
            student_model.train()
            stage_losses = []
            
            # Combine with previous stages for continued learning
            if stage_idx > 0:
                prev_X = torch.cat([stages[0] for stages in curriculum_stages[:stage_idx]], dim=0)
                prev_y = torch.cat([stages[1] for stages in curriculum_stages[:stage_idx]], dim=0)
                combined_X = torch.cat([prev_X, stage_X], dim=0)
                combined_y = torch.cat([prev_y, stage_y], dim=0)
            else:
                combined_X, combined_y = stage_X, stage_y
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(combined_X, combined_y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
            
            # Train on current stage
            for epoch in range(self.stage_epochs):
                epoch_loss = 0.0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Student prediction
                    student_pred = student_model(batch_X)
                    
                    # Teacher ensemble prediction
                    teacher_preds = []
                    for teacher in teacher_ensemble:
                        teacher.eval()
                        with torch.no_grad():
                            teacher_pred = teacher(batch_X)
                            if temperature != 1.0:
                                teacher_pred = teacher_pred / temperature
                            teacher_preds.append(teacher_pred)
                    
                    ensemble_pred = torch.stack(teacher_preds, dim=0).mean(dim=0)
                    
                    # Distillation loss
                    if y_train.dtype == torch.long:
                        # Classification
                        distill_loss = F.kl_div(
                            F.log_softmax(student_pred / temperature, dim=1),
                            F.softmax(ensemble_pred, dim=1),
                            reduction='batchmean'
                        )
                        task_loss = F.cross_entropy(student_pred, batch_y)
                    else:
                        # Regression
                        distill_loss = F.mse_loss(student_pred.squeeze(), ensemble_pred)
                        task_loss = F.mse_loss(student_pred.squeeze(), batch_y)
                    
                    # Progressive weight (more task loss in later stages)
                    alpha = 0.8 - 0.3 * (stage_idx / (self.n_stages - 1))  # 0.8 -> 0.5
                    loss = alpha * distill_loss + (1 - alpha) * task_loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                stage_losses.append(epoch_loss / len(dataloader))
            
            all_losses.extend(stage_losses)
            
            # Evaluate stage performance
            student_model.eval()
            with torch.no_grad():
                stage_pred = student_model(stage_X)
                if stage_y.dtype == torch.long:
                    stage_acc = accuracy_score(stage_y.numpy(), stage_pred.argmax(dim=1).numpy())
                    stage_performances.append({'stage': stage_idx, 'accuracy': stage_acc})
                else:
                    stage_mse = mean_squared_error(stage_y.numpy(), stage_pred.squeeze().numpy())
                    stage_performances.append({'stage': stage_idx, 'mse': stage_mse})
            
            print(f"Stage {stage_idx + 1} completed. Performance: {stage_performances[-1]}")
        
        return {
            'total_losses': all_losses,
            'stage_performances': stage_performances
        }
    
    def evaluate_compression(
        self,
        teacher_ensemble: List[nn.Module],
        student_model: nn.Module,
        test_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate progressive distillation results"""
        
        # Use standard evaluation
        standard_distiller = StandardDistillation()
        return standard_distiller.evaluate_compression(teacher_ensemble, student_model, test_data)


class FinancialEnsembleDistiller:
    """
    Financial-specific ensemble distillation
    """
    
    def __init__(
        self,
        distillation_method: str = 'standard',
        financial_features: Optional[List[str]] = None
    ):
        self.distillation_method = distillation_method
        self.financial_features = financial_features or [
            'price_features', 'technical_indicators', 'volume_features', 'volatility_features'
        ]
        
        self.distiller = self._create_distiller()
        self.feature_scalers = {}
        
    def _create_distiller(self) -> EnsembleDistiller:
        """Create appropriate distiller"""
        
        if self.distillation_method == 'standard':
            return StandardDistillation(alpha=0.6, beta=0.4)
        elif self.distillation_method == 'attention':
            return AttentionDistillation()
        elif self.distillation_method == 'progressive':
            return ProgressiveDistillation(n_stages=4)
        else:
            raise ValueError(f"Unknown distillation method: {self.distillation_method}")
    
    def create_teacher_ensemble(
        self,
        market_data: pd.DataFrame,
        ensemble_types: List[str] = ['lstm', 'transformer', 'cnn', 'mlp'],
        window_size: int = 20
    ) -> List[nn.Module]:
        """Create diverse teacher ensemble for financial data"""
        
        # Calculate input dimension
        n_features = len(self.financial_features)
        input_dim = window_size * n_features
        
        teachers = []
        
        for model_type in ensemble_types:
            if model_type == 'lstm':
                teacher = self._create_lstm_teacher(n_features, 64, 1)
            elif model_type == 'transformer':
                teacher = self._create_transformer_teacher(n_features, 64, 1)
            elif model_type == 'cnn':
                teacher = self._create_cnn_teacher(window_size, n_features, 1)
            elif model_type == 'mlp':
                teacher = self._create_mlp_teacher(input_dim, [128, 64], 1)
            else:
                continue
            
            teachers.append(teacher)
        
        return teachers
    
    def _create_lstm_teacher(self, input_size: int, hidden_size: int, output_size: int) -> nn.Module:
        """Create LSTM teacher model"""
        
        class LSTMTeacher(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                # Reshape for LSTM: (batch, seq_len, features)
                batch_size = x.size(0)
                seq_len = x.size(1) // input_size
                x = x.view(batch_size, seq_len, input_size)
                
                lstm_out, _ = self.lstm(x)
                out = self.dropout(lstm_out[:, -1, :])  # Last time step
                return self.fc(out)
        
        return LSTMTeacher(input_size, hidden_size, output_size)
    
    def _create_transformer_teacher(self, input_size: int, d_model: int, output_size: int) -> nn.Module:
        """Create Transformer teacher model"""
        
        class TransformerTeacher(nn.Module):
            def __init__(self, input_size, d_model, output_size):
                super().__init__()
                self.input_proj = nn.Linear(input_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.fc = nn.Linear(d_model, output_size)
                
            def forward(self, x):
                batch_size = x.size(0)
                seq_len = x.size(1) // input_size
                x = x.view(batch_size, seq_len, input_size)
                
                x = self.input_proj(x)
                transformer_out = self.transformer(x)
                out = transformer_out.mean(dim=1)  # Global average pooling
                return self.fc(out)
        
        return TransformerTeacher(input_size, d_model, output_size)
    
    def _create_cnn_teacher(self, seq_len: int, input_size: int, output_size: int) -> nn.Module:
        """Create CNN teacher model"""
        
        class CNNTeacher(nn.Module):
            def __init__(self, seq_len, input_size, output_size):
                super().__init__()
                self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(32, output_size)
                
            def forward(self, x):
                batch_size = x.size(0)
                x = x.view(batch_size, input_size, seq_len)
                
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = self.pool(x).squeeze(-1)
                return self.fc(x)
        
        return CNNTeacher(seq_len, input_size, output_size)
    
    def _create_mlp_teacher(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
        """Create MLP teacher model"""
        
        class MLPTeacher(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim):
                super().__init__()
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, output_dim))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return MLPTeacher(input_dim, hidden_dims, output_dim)
    
    def prepare_financial_data(
        self,
        market_data: pd.DataFrame,
        target_column: str = 'return',
        window_size: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare financial data for distillation"""
        
        # Add financial features
        data = market_data.copy()
        
        # Basic features
        data['return'] = data['close'].pct_change()
        data['log_return'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['return'].rolling(10).std()
        
        # Technical indicators
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_10'] = data['close'].rolling(10).mean()
        data['rsi'] = self._calculate_rsi(data['close'])
        
        # Volume features
        data['volume_sma'] = data['volume'].rolling(10).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Create windows
        features = []
        targets = []
        
        feature_cols = ['return', 'log_return', 'volatility', 'rsi', 'volume_ratio']
        
        for i in range(window_size, len(data)):
            # Feature window
            window = data.iloc[i-window_size:i][feature_cols].values
            features.append(window.flatten())
            
            # Target
            if target_column == 'direction':
                target = 1 if data.iloc[i]['close'] > data.iloc[i-1]['close'] else 0
            else:
                target = data.iloc[i][target_column] if target_column in data.columns else data.iloc[i]['return']
            
            targets.append(target if not np.isnan(target) else 0.0)
        
        X = torch.FloatTensor(np.array(features))
        y = torch.LongTensor(targets) if target_column == 'direction' else torch.FloatTensor(targets)
        
        return X, y
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def distill_financial_ensemble(
        self,
        market_data: pd.DataFrame,
        student_model: nn.Module,
        target_column: str = 'return',
        window_size: int = 20,
        n_epochs: int = 100
    ) -> Dict[str, Any]:
        """Distill financial ensemble to student model"""
        
        # Prepare data
        X, y = self.prepare_financial_data(market_data, target_column, window_size)
        
        # Create teacher ensemble
        teacher_ensemble = self.create_teacher_ensemble(market_data, window_size=window_size)
        
        # Pre-train teachers (simplified - in practice would train properly)
        print("Pre-training teacher ensemble...")
        for i, teacher in enumerate(teacher_ensemble):
            print(f"Training teacher {i+1}/{len(teacher_ensemble)}")
            optimizer = optim.Adam(teacher.parameters(), lr=0.001)
            
            # Quick training
            for epoch in range(20):
                optimizer.zero_grad()
                pred = teacher(X)
                
                if y.dtype == torch.long:
                    loss = F.cross_entropy(pred, y)
                else:
                    loss = F.mse_loss(pred.squeeze(), y)
                
                loss.backward()
                optimizer.step()
        
        # Distillation
        print("Starting knowledge distillation...")
        distill_results = self.distiller.distill(
            teacher_ensemble, student_model, (X, y), n_epochs=n_epochs
        )
        
        # Evaluation
        test_split = int(0.8 * len(X))
        X_test, y_test = X[test_split:], y[test_split:]
        
        eval_results = self.distiller.evaluate_compression(
            teacher_ensemble, student_model, (X_test, y_test)
        )
        
        return {
            'distillation_results': distill_results,
            'evaluation_results': eval_results,
            'teacher_ensemble_size': len(teacher_ensemble)
        }


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Testing Ensemble Distillation Methods...")
    
    # Create test models
    class TestModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
            self.fc3 = nn.Linear(hidden_dim//2, output_dim)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    # Test data
    input_dim = 50
    n_classes = 3
    n_samples = 1000
    
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    
    # Create teacher ensemble
    teacher_ensemble = [
        TestModel(input_dim, 128, n_classes),
        TestModel(input_dim, 96, n_classes),
        TestModel(input_dim, 64, n_classes)
    ]
    
    # Create student model (smaller)
    student_model = TestModel(input_dim, 32, n_classes)
    
    # Test Standard Distillation
    print("\nTesting Standard Distillation...")
    try:
        standard_distiller = StandardDistillation(alpha=0.7, beta=0.3)
        
        results = standard_distiller.distill(
            teacher_ensemble, student_model, (X[:800], y[:800]), n_epochs=50
        )
        
        eval_results = standard_distiller.evaluate_compression(
            teacher_ensemble, student_model, (X[800:], y[800:])
        )
        
        print(f"Compression ratio: {eval_results['compression_ratio']:.2f}")
        print(f"Accuracy retention: {eval_results['accuracy_retention']:.3f}")
        
    except Exception as e:
        print(f"Standard distillation error: {e}")
    
    # Test Financial Ensemble Distillation
    print("\nTesting Financial Ensemble Distillation...")
    try:
        # Generate synthetic financial data
        n_days = 1000
        base_price = 100
        
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = base_price * np.cumprod(1 + returns)
        
        financial_df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_days))),
            'close': prices,
            'volume': np.random.exponential(1000000, n_days)
        })
        
        # Create financial distiller
        financial_distiller = FinancialEnsembleDistiller(distillation_method='standard')
        
        # Create compact student model
        financial_student = TestModel(input_dim=100, hidden_dim=64, output_dim=1)  # 20 window * 5 features
        
        # Distill ensemble
        financial_results = financial_distiller.distill_financial_ensemble(
            financial_df, financial_student, target_column='return', n_epochs=30
        )
        
        print("Financial Distillation Results:")
        eval_res = financial_results['evaluation_results']
        print(f"  Teacher Parameters: {eval_res['teacher_parameters']}")
        print(f"  Student Parameters: {eval_res['student_parameters']}")
        print(f"  Compression Ratio: {eval_res['compression_ratio']:.2f}")
        
        if 'r2_retention' in eval_res:
            print(f"  R² Retention: {eval_res['r2_retention']:.3f}")
        
    except Exception as e:
        print(f"Financial ensemble distillation error: {e}")
    
    print("\nDone!")