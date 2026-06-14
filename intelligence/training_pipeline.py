"""
QUANTUM-FORGE: ML Model Training Pipeline
============================================
P0 — Without trained models, 40% of signal fusion is random noise.

This module:
1. Loads historical data from Parquet (collected by historical_collector.py)
2. Creates training datasets with proper features/labels
3. Trains each model type (LSTM, Transformer, TCN, etc.) 
4. Saves trained weights to intelligence/trained_models/
5. Validates models with walk-forward backtesting
6. Reports training metrics

Training approach:
  - Supervised: Predict next-period return direction/magnitude
  - Walk-forward: Train on [0:T], validate on [T:T+V], slide forward
  - Labels: Next-period return clipped to [-1, 1]
  - Loss: MSE for regression models, CE for classification
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("MLTrainer")

MODELS_DIR = Path("./intelligence/trained_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class FeatureEngineer:
    """
    Creates features from OHLCV data for model training.
    Produces the same 20-feature vector as MLEnsembleEngine.extract_features()
    but operates on full DataFrames for batch processing.
    """
    
    FEATURE_DIM = 32  # Aligned with core.feature_pipeline (32 features)
    LOOKBACK = 50  # Need at least 50 periods of history for all features
    
    @staticmethod
    def compute_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute feature matrix and target labels from OHLCV data.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            features: (N, 20) array of features
            targets: (N,) array of next-period returns
        """
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float) if 'volume' in df.columns else None
        high = df['high'].values.astype(float) if 'high' in df.columns else close
        low = df['low'].values.astype(float) if 'low' in df.columns else close
        
        returns = np.diff(close) / close[:-1]
        
        features_list = []
        targets_list = []
        
        lookback = FeatureEngineer.LOOKBACK
        
        for i in range(lookback, len(close) - 1):
            feature_vec = FeatureEngineer._extract_single(
                close[:i+1], returns[:i], volume[:i+1] if volume is not None else None,
                high[:i+1], low[:i+1]
            )
            features_list.append(feature_vec)
            
            # Target: next-period return, clipped
            next_ret = (close[i+1] - close[i]) / close[i]
            targets_list.append(np.clip(next_ret * 10, -1, 1))  # Scale up, clip to [-1, 1]
        
        if not features_list:
            return np.array([]), np.array([])
        
        return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)
    
    @staticmethod
    def _extract_single(
        prices: np.ndarray,
        returns: np.ndarray,
        volumes: Optional[np.ndarray],
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> np.ndarray:
        """Extract a single feature vector — uses FeaturePipeline for 32 features."""
        try:
            from core.feature_pipeline import FeaturePipeline
            pipe = FeaturePipeline(feature_dim=FeatureEngineer.FEATURE_DIM)
            return pipe.extract(prices=prices, volumes=volumes)
        except ImportError:
            pass
        # Fallback if FeaturePipeline unavailable:
        features = []
        
        # 1. Multi-timeframe returns (5)
        for lb in [1, 3, 5, 10, 20]:
            if len(returns) >= lb:
                features.append(np.mean(returns[-lb:]))
            else:
                features.append(0.0)
        
        # 2. Multi-timeframe volatility (4)
        for lb in [5, 10, 20, 50]:
            if len(returns) >= lb:
                features.append(np.std(returns[-lb:]))
            else:
                features.append(0.0)
        
        # 3. Z-scores (3)
        for window in [10, 20, 50]:
            if len(prices) >= window:
                mu = np.mean(prices[-window:])
                sigma = np.std(prices[-window:])
                features.append((prices[-1] - mu) / sigma if sigma > 0 else 0.0)
            else:
                features.append(0.0)
        
        # 4. RSI-like (1)
        if len(returns) >= 14:
            gains = np.maximum(returns[-14:], 0)
            losses = np.abs(np.minimum(returns[-14:], 0))
            avg_gain, avg_loss = np.mean(gains), np.mean(losses)
            rsi = (1.0 - 1.0 / (1.0 + avg_gain / avg_loss)) if avg_loss > 0 else 1.0
            features.append(rsi - 0.5)
        else:
            features.append(0.0)
        
        # 5. MACD-like (1)
        if len(prices) >= 26:
            ema12 = np.mean(prices[-12:])
            ema26 = np.mean(prices[-26:])
            features.append(np.clip((ema12 - ema26) / ema26 * 100, -1, 1))
        else:
            features.append(0.0)
        
        # 6. ROC (1)
        if len(prices) >= 10:
            roc = (prices[-1] - prices[-10]) / prices[-10]
            features.append(np.clip(roc * 10, -1, 1))
        else:
            features.append(0.0)
        
        # 7. Volume features (2)
        if volumes is not None and len(volumes) >= 10:
            vol_ratio = np.mean(volumes[-3:]) / (np.mean(volumes[-10:]) + 1e-10)
            features.append(np.clip(vol_ratio - 1.0, -1, 1))
            vol_trend = np.mean(volumes[-5:]) - np.mean(volumes[-10:-5])
            features.append(np.clip(vol_trend / (np.mean(volumes[-10:]) + 1e-10), -1, 1))
        else:
            features.extend([0.0, 0.0])
        
        # 8. Autocorrelation (1)
        if len(returns) >= 20:
            ac = np.corrcoef(returns[:-1][-19:], returns[1:][-19:])[0, 1]
            features.append(ac if not np.isnan(ac) else 0.0)
        else:
            features.append(0.0)
        
        # 9. Skewness and Kurtosis (2)
        if len(returns) >= 20:
            from scipy import stats as sp_stats
            features.append(np.clip(sp_stats.skew(returns[-20:]) / 3, -1, 1))
            features.append(np.clip((sp_stats.kurtosis(returns[-20:]) - 3) / 10, -1, 1))
        else:
            features.extend([0.0, 0.0])
        
        arr = np.array(features[:FeatureEngineer.FEATURE_DIM], dtype=np.float32)
        if len(arr) < FeatureEngineer.FEATURE_DIM:
            arr = np.pad(arr, (0, FeatureEngineer.FEATURE_DIM - len(arr)))
        
        return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)


class ModelTrainer:
    """
    Trains a single PyTorch model on financial features.
    
    Walk-forward training:
        [=====TRAIN=====][==VAL==] → slide → [=====TRAIN=====][==VAL==]
    """
    
    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs_per_fold: int = 30,
        early_stop_patience: int = 5,
    ):
        self.model_name = model_name
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epochs_per_fold = epochs_per_fold
        self.early_stop_patience = early_stop_patience
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.loss_fn = nn.MSELoss()
        
        self.training_history: List[Dict] = []
    
    def train_walk_forward(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        n_folds: int = 5,
        train_ratio: float = 0.8,
    ) -> Dict:
        """
        Walk-forward training with multiple folds.
        
        Returns:
            Training metrics dict
        """
        n_samples = len(features)
        fold_size = n_samples // n_folds
        
        all_val_losses = []
        all_val_accuracies = []
        
        for fold in range(n_folds):
            # Calculate fold boundaries
            train_end = int((fold + 1) * fold_size * train_ratio)
            val_start = train_end
            val_end = min(val_start + int(fold_size * (1 - train_ratio)), n_samples)
            
            if val_start >= n_samples or val_end <= val_start:
                continue
            
            train_start = max(0, train_end - fold_size)
            
            X_train = features[train_start:train_end]
            y_train = targets[train_start:train_end]
            X_val = features[val_start:val_end]
            y_val = targets[val_start:val_end]
            
            if len(X_train) < 100 or len(X_val) < 20:
                continue
            
            logger.info(f"  Fold {fold+1}/{n_folds}: train={len(X_train)}, val={len(X_val)}")
            
            fold_metrics = self._train_fold(X_train, y_train, X_val, y_val)
            all_val_losses.append(fold_metrics['best_val_loss'])
            all_val_accuracies.append(fold_metrics['val_accuracy'])
            
            self.training_history.append({
                'fold': fold,
                **fold_metrics,
            })
        
        avg_val_loss = np.mean(all_val_losses) if all_val_losses else float('inf')
        avg_val_acc = np.mean(all_val_accuracies) if all_val_accuracies else 0.0
        
        return {
            'model_name': self.model_name,
            'n_folds': len(all_val_losses),
            'avg_val_loss': float(avg_val_loss),
            'avg_val_accuracy': float(avg_val_acc),
            'total_samples': n_samples,
        }
    
    def _train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        """Train a single fold with early stopping."""
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).unsqueeze(1),  # (N, 1, features) for sequence models
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
        y_val_t = torch.FloatTensor(y_val)
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        self.model.train()
        
        for epoch in range(self.epochs_per_fold):
            # Train
            train_loss = 0.0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                
                try:
                    output = self.model(X_batch)
                    if output.dim() > 1:
                        output = output.squeeze(-1)
                    if output.dim() > 1:
                        output = output[:, -1]  # Take last timestep
                    
                    loss = self.loss_fn(output, y_batch)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    n_batches += 1
                except Exception as e:
                    logger.debug(f"Training batch error: {e}")
                    continue
            
            if n_batches == 0:
                continue
                
            avg_train_loss = train_loss / n_batches
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                try:
                    val_output = self.model(X_val_t)
                    if val_output.dim() > 1:
                        val_output = val_output.squeeze(-1)
                    if val_output.dim() > 1:
                        val_output = val_output[:, -1]
                    
                    val_loss = self.loss_fn(val_output, y_val_t).item()
                    
                    # Directional accuracy
                    pred_dir = torch.sign(val_output)
                    actual_dir = torch.sign(y_val_t)
                    accuracy = (pred_dir == actual_dir).float().mean().item()
                except:
                    val_loss = float('inf')
                    accuracy = 0.0
            
            self.model.train()
            
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    logger.info(f"    Early stop at epoch {epoch+1}")
                    break
        
        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Final validation accuracy
        self.model.eval()
        with torch.no_grad():
            try:
                val_output = self.model(X_val_t)
                if val_output.dim() > 1:
                    val_output = val_output.squeeze(-1)
                if val_output.dim() > 1:
                    val_output = val_output[:, -1]
                pred_dir = torch.sign(val_output)
                actual_dir = torch.sign(y_val_t)
                final_accuracy = (pred_dir == actual_dir).float().mean().item()
            except:
                final_accuracy = 0.0
        
        return {
            'best_val_loss': best_val_loss,
            'val_accuracy': final_accuracy,
            'epochs_trained': epoch + 1 if 'epoch' in dir() else 0,
        }
    
    def save(self, path: Optional[Path] = None):
        """Save trained model weights."""
        if path is None:
            path = MODELS_DIR / f"{self.model_name}.pt"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
        }, path)
        
        logger.info(f"Saved {self.model_name} weights to {path}")
    
    def load(self, path: Optional[Path] = None) -> bool:
        """Load previously saved model weights.
        
        Returns:
            True if weights were loaded successfully, False otherwise.
        """
        if path is None:
            path = MODELS_DIR / f"{self.model_name}.pt"
        
        if not path.exists():
            logger.warning(f"No saved weights found at {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.training_history = checkpoint.get('training_history', [])
            logger.info(
                f"Loaded {self.model_name} weights from {path} "
                f"(saved {checkpoint.get('timestamp', 'unknown')})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load weights from {path}: {e}")
            return False


class TrainingPipeline:
    """
    End-to-end training pipeline for all ML models.
    
    Usage:
        pipeline = TrainingPipeline()
        pipeline.run(symbols=["BTCUSDT", "ETHUSDT"], days=90)
    """
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict] = {}
    
    def run(
        self,
        symbols: List[str] = None,
        days: int = 90,
        interval: str = "1h",
        n_folds: int = 5,
        epochs: int = 30,
    ) -> Dict:
        """
        Run the full training pipeline.
        
        Args:
            symbols: Symbols to train on
            days: Days of historical data
            interval: Candle interval
            n_folds: Walk-forward folds
            epochs: Max epochs per fold
        
        Returns:
            Training results dict
        """
        symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        
        logger.info("=" * 80)
        logger.info("QUANTUM-FORGE ML TRAINING PIPELINE")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Days: {days}, Interval: {interval}, Folds: {n_folds}")
        logger.info("=" * 80)
        
        # Step 1: Load data
        logger.info("[STEP 1] Loading historical data...")
        all_features, all_targets = self._load_training_data(symbols, interval, days)
        
        if len(all_features) < 500:
            logger.error(f"Insufficient training data: {len(all_features)} samples (need 500+)")
            logger.info("Run 'python data/historical_collector.py --days 90' first")
            return {'error': 'insufficient_data', 'samples': len(all_features)}
        
        logger.info(f"  Total samples: {len(all_features)}")
        logger.info(f"  Feature dim: {all_features.shape[1]}")
        logger.info(f"  Target range: [{all_targets.min():.3f}, {all_targets.max():.3f}]")
        
        # Step 2: Train each model
        models_to_train = self._create_models()
        
        for model_name, model in models_to_train.items():
            logger.info(f"\n[TRAINING] {model_name}...")
            
            try:
                trainer = ModelTrainer(
                    model_name=model_name,
                    model=model,
                    epochs_per_fold=epochs,
                )
                
                result = trainer.train_walk_forward(
                    all_features, all_targets,
                    n_folds=n_folds,
                )
                
                # Save
                trainer.save(self.models_dir / f"{model_name}.pt")
                
                self.results[model_name] = result
                logger.info(
                    f"  {model_name}: val_loss={result['avg_val_loss']:.4f}, "
                    f"accuracy={result['avg_val_accuracy']:.1%}"
                )
                
            except Exception as e:
                logger.error(f"  Failed to train {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
        
        # Step 3: Save training report
        self._save_report()
        
        # Step 4: Print summary
        self._print_summary()
        
        return self.results
    
    def _load_training_data(
        self,
        symbols: List[str],
        interval: str,
        days: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and combine training data from all symbols."""
        from data.historical_collector import HistoricalDataCollector
        
        collector = HistoricalDataCollector(symbols=symbols, intervals=[interval])
        
        all_features = []
        all_targets = []
        
        for symbol in symbols:
            df = collector.load_ohlcv(symbol, interval, days)
            
            if df.empty or len(df) < 100:
                logger.warning(f"  Skipping {symbol}: only {len(df)} candles")
                continue
            
            features, targets = FeatureEngineer.compute_features(df)
            
            if len(features) > 0:
                all_features.append(features)
                all_targets.append(targets)
                logger.info(f"  {symbol}: {len(features)} samples from {len(df)} candles")
        
        if not all_features:
            return np.array([]), np.array([])
        
        return np.vstack(all_features), np.concatenate(all_targets)
    
    def _create_models(self) -> Dict[str, nn.Module]:
        """Create fresh model instances for training."""
        models = {}
        feature_dim = FeatureEngineer.FEATURE_DIM
        
        # LSTM
        try:
            from intelligence.deep_learning.deep_learning_models import LSTMModel
            models['lstm'] = LSTMModel(
                input_dim=feature_dim, hidden_dim=64, num_layers=2, output_dim=1
            )
        except Exception as e:
            logger.warning(f"Can't create LSTM: {e}")
        
        # GRU
        try:
            from intelligence.deep_learning.deep_learning_models import GRUModel
            models['gru'] = GRUModel(
                input_dim=feature_dim, hidden_dim=64, num_layers=2, output_dim=1
            )
        except Exception as e:
            logger.warning(f"Can't create GRU: {e}")
        
        # Transformer
        try:
            from intelligence.deep_learning.deep_learning_models import TransformerModel
            models['transformer'] = TransformerModel(
                input_dim=feature_dim, hidden_dim=64, num_heads=4, num_layers=2
            )
        except Exception as e:
            logger.warning(f"Can't create Transformer: {e}")
        
        # TCN
        try:
            from intelligence.deep_learning.temporal_models import TemporalConvNet as TCN
            models['tcn'] = TCN(input_channels=feature_dim, output_size=1)
        except Exception as e:
            logger.warning(f"Can't create TCN: {e}")
        
        # Simple MLP (always works)
        class TradingMLP(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Tanh(),
                )
            def forward(self, x):
                if x.dim() == 3:
                    x = x[:, -1, :]  # Take last timestep
                return self.net(x)
        
        models['mlp'] = TradingMLP(feature_dim)
        
        logger.info(f"Created {len(models)} models for training: {list(models.keys())}")
        return models
    
    def _save_report(self):
        """Save training report to disk."""
        report_path = self.models_dir / "training_report.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': {},
        }
        
        for name, r in self.results.items():
            # Clean for JSON serialization
            clean = {}
            for k, v in r.items():
                if isinstance(v, (float, np.floating)):
                    clean[k] = float(v)
                elif isinstance(v, (int, np.integer)):
                    clean[k] = int(v)
                else:
                    clean[k] = v
            report['results'][name] = clean
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")
    
    def _print_summary(self):
        """Print training summary."""
        print("\n" + "=" * 70)
        print("ML TRAINING PIPELINE — RESULTS")
        print("=" * 70)
        
        for name, r in self.results.items():
            if 'error' in r:
                print(f"  {name:20s}: FAILED — {r['error']}")
            else:
                print(
                    f"  {name:20s}: loss={r['avg_val_loss']:.4f}, "
                    f"accuracy={r['avg_val_accuracy']:.1%}, "
                    f"folds={r['n_folds']}"
                )
        
        print("=" * 70)
        print(f"Weights saved to: {self.models_dir}")
        print("These will be auto-loaded by MLEnsembleEngine on next start.")
        print("=" * 70)


def main():
    """CLI entry point for model training."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(message)s',
    )
    
    parser = argparse.ArgumentParser(description='Quantum-Forge ML Training Pipeline')
    parser.add_argument('--symbols', nargs='+', default=None)
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--interval', type=str, default='1h')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    pipeline.run(
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        n_folds=args.folds,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
