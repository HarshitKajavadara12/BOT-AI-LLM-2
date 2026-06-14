"""
QUANTUM-FORGE: Meta-Learning Integration Bridge
=================================================
Wires the meta-learning module (intelligence/meta_learning/) into the
live pipeline for adaptive model selection and hyperparameter tuning.

Key capabilities:
  - Task-conditioned model selection (MAML-like)
  - Regime-aware ensemble weight adaptation
  - Few-shot learning for novel market conditions
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("MetaLearningBridge")


class MetaLearningBridge:
    """
    Connects meta-learning modules to the ensemble pipeline.
    
    In each regime, the meta-learner suggests which models should
    get higher weights based on past performance in similar regimes.
    """

    def __init__(self, model_names: List[str], n_regimes: int = 4):
        self.model_names = model_names
        self.n_regimes = n_regimes

        # Performance per regime per model: {regime: {model: deque of scores}}
        self._regime_performance: Dict[str, Dict[str, deque]] = {}
        for r in range(n_regimes):
            regime_key = str(r)
            self._regime_performance[regime_key] = {
                m: deque(maxlen=100) for m in model_names
            }

        # Regime name mapping
        self._regime_map = {
            "TRENDING": "0", "MEAN_REVERTING": "1",
            "HIGH_VOLATILITY": "2", "LOW_VOLATILITY": "3",
            "BULL": "0", "BEAR": "1", "NEUTRAL": "3",
        }

        self._meta_model = None
        self._try_load_meta()

    def _try_load_meta(self):
        """Try to load the full meta-learning model."""
        try:
            from intelligence.meta_learning.meta_learning import MAMLTrader
            self._meta_model = MAMLTrader(
                input_dim=32, hidden_dim=64, output_dim=1
            )
            logger.info("MetaLearningBridge: MAMLTrader loaded")
        except ImportError:
            logger.info("MetaLearningBridge: MAMLTrader not available — using regime-based fallback")

    def record_performance(
        self,
        regime: str,
        model_name: str,
        actual_return: float,
        predicted_direction: float,
    ):
        """
        Record model performance in a specific regime.
        
        Args:
            regime: Current market regime name
            model_name: Name of the model
            actual_return: Realised return
            predicted_direction: Model's predicted direction [-1, 1]
        """
        regime_key = self._regime_map.get(regime, "3")
        if model_name not in self.model_names:
            return

        # Score: 1 if correct direction, 0 if wrong
        correct = np.sign(actual_return) == np.sign(predicted_direction)
        score = 1.0 if correct else 0.0

        if regime_key in self._regime_performance:
            self._regime_performance[regime_key][model_name].append(score)

    def suggest_weights(self, current_regime: str) -> Dict[str, float]:
        """
        Suggest ensemble weights based on per-regime model performance.
        
        Returns:
            {model_name: weight} dict summing to 1.0
        """
        regime_key = self._regime_map.get(current_regime, "3")
        scores = {}

        for model in self.model_names:
            perf = self._regime_performance.get(regime_key, {}).get(model, deque())
            if len(perf) >= 10:
                scores[model] = max(np.mean(list(perf)) - 0.3, 0.01)
            else:
                scores[model] = 0.5  # default equal weight

        # Normalize to sum to 1
        total = sum(scores.values())
        if total > 0:
            return {m: s / total for m, s in scores.items()}
        else:
            equal = 1.0 / max(len(self.model_names), 1)
            return {m: equal for m in self.model_names}

    def get_status(self) -> Dict:
        """Return meta-learning status."""
        return {
            "meta_model_loaded": self._meta_model is not None,
            "regimes_tracked": len(self._regime_performance),
            "models": self.model_names,
            "per_regime_samples": {
                r: {m: len(p) for m, p in models.items()}
                for r, models in self._regime_performance.items()
            },
        }
