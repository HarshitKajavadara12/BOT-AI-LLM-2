"""
Test Suite for Machine Learning Models

Tests against actual classes in the intelligence/ package with correct
constructor signatures verified from source code.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gymnasium as gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False


# ============================================================================
# Reinforcement Learning
# ============================================================================

class TestPPOAgent(unittest.TestCase):
    """Test PPOAgent (reinforcement_learning/ppo_agent.py)."""

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_instantiate(self):
        from intelligence.reinforcement_learning.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=20, action_dim=3)
        self.assertIsNotNone(agent)


class TestSACAgent(unittest.TestCase):
    """Test SACAgent (actual class name in soft_actor_critic.py)."""

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_instantiate(self):
        from intelligence.reinforcement_learning.soft_actor_critic import SACAgent
        agent = SACAgent(state_dim=20, action_dim=3)
        self.assertIsNotNone(agent)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_custom_hyperparams(self):
        from intelligence.reinforcement_learning.soft_actor_critic import SACAgent
        agent = SACAgent(state_dim=10, action_dim=2, lr=1e-3, gamma=0.95, tau=0.01)
        self.assertIsNotNone(agent)


class TestMarketEnvironment(unittest.TestCase):
    """Test MarketEnvironment (reinforcement_learning/market_env.py).
    
    Requires gym and pd.DataFrame with 'close' column.
    """

    @unittest.skipIf(not GYM_AVAILABLE, "gym/gymnasium not available")
    def test_instantiate(self):
        from intelligence.reinforcement_learning.market_env import MarketEnvironment
        df = pd.DataFrame({
            'close': np.random.uniform(90, 110, 200),
            'volume': np.random.uniform(1000, 5000, 200),
        })
        env = MarketEnvironment(data=df)
        self.assertIsNotNone(env)


class TestDQNAgent(unittest.TestCase):
    """Test DQNAgent from reinforcement_learning.py.
    
    Requires AgentConfig dataclass with all fields specified.
    """

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_instantiate(self):
        from intelligence.reinforcement_learning.reinforcement_learning import (
            DQNAgent, AgentConfig, AgentType, ActionSpace
        )
        config = AgentConfig(
            agent_type=AgentType.DQN,
            state_dim=20,
            action_dim=3,
            action_space=ActionSpace.DISCRETE,
            hidden_dim=128,
            learning_rate=1e-3,
            gamma=0.99,
            tau=0.005,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            buffer_size=10000,
            batch_size=64,
            update_frequency=4,
            target_update_frequency=100,
        )
        agent = DQNAgent(config)
        self.assertIsNotNone(agent)


# ============================================================================
# Meta-Learning
# ============================================================================

class TestMAML(unittest.TestCase):
    """Test MAML (meta_learning/meta_learning.py).
    
    Requires MetaModel + MetaLearningConfig.
    """

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_instantiate(self):
        from intelligence.meta_learning.meta_learning import (
            MAML, MetaModel, MetaLearningConfig, MetaLearningType, TaskType
        )
        model = MetaModel(input_dim=20, output_dim=1, hidden_dim=64, num_layers=2)
        config = MetaLearningConfig(
            algorithm_type=MetaLearningType.MAML,
            task_type=TaskType.REGRESSION,
            input_dim=20,
            output_dim=1,
            hidden_dim=64,
            num_layers=2,
            meta_learning_rate=1e-3,
            inner_learning_rate=0.01,
            num_inner_steps=5,
            num_meta_epochs=100,
            meta_batch_size=4,
            support_size=10,
            query_size=10,
            num_ways=5,
        )
        maml = MAML(model=model, config=config)
        self.assertIsNotNone(maml)


class TestFewShotLearner(unittest.TestCase):
    """Test FewShotLearner base class."""

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_import(self):
        from intelligence.meta_learning.few_shot_learning import FewShotLearner
        self.assertIsNotNone(FewShotLearner)


# ============================================================================
# Probabilistic ML
# ============================================================================

class TestGaussianProcess(unittest.TestCase):
    """Test GaussianProcess from probabilistic_ml.py."""

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_import(self):
        try:
            from intelligence.probabilistic_ml.probabilistic_ml import GaussianProcess
            self.assertIsNotNone(GaussianProcess)
        except ImportError:
            self.skipTest("gpytorch not available")


class TestFinancialGaussianProcess(unittest.TestCase):
    """Test FinancialGaussianProcess (gaussian_processes.py)."""

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_import(self):
        from intelligence.probabilistic_ml.gaussian_processes import FinancialGaussianProcess, GPYTORCH_AVAILABLE
        self.assertIsNotNone(FinancialGaussianProcess)
        # Class exists; full instantiation requires gpytorch + training data
        self.assertIsInstance(GPYTORCH_AVAILABLE, bool)


class TestVariationalAutoencoder(unittest.TestCase):
    """Test VariationalAutoencoder (probabilistic_ml/variational_inference.py)."""

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_instantiate(self):
        from intelligence.probabilistic_ml.variational_inference import VariationalAutoencoder
        vae = VariationalAutoencoder(input_dim=20, latent_dim=5)
        self.assertIsNotNone(vae)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_forward(self):
        import torch
        from intelligence.probabilistic_ml.variational_inference import VariationalAutoencoder
        vae = VariationalAutoencoder(input_dim=20, latent_dim=5)
        x = torch.randn(8, 20)
        result = vae(x)
        self.assertIsNotNone(result)


class TestBayesianOptimizer(unittest.TestCase):
    """Test BayesianOptimizer (bayesian_optimization.py).
    
    Requires ObjectiveFunction subclass.
    """

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_import(self):
        try:
            from intelligence.probabilistic_ml.bayesian_optimization import BayesianOptimizer
            self.assertIsNotNone(BayesianOptimizer)
        except ImportError:
            self.skipTest("botorch not available")


# ============================================================================
# Feature Learning
# ============================================================================

class TestFeatureLearningModule(unittest.TestCase):
    """Test feature_learning.py classes."""

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_denoising_autoencoder_import(self):
        from intelligence.feature_learning.feature_learning import DenoisingAutoencoder
        self.assertIsNotNone(DenoisingAutoencoder)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_sparse_autoencoder_import(self):
        from intelligence.feature_learning.feature_learning import SparseAutoencoder
        self.assertIsNotNone(SparseAutoencoder)


class TestRepresentationLearner(unittest.TestCase):
    """Test RepresentationLearner base class."""

    def test_import(self):
        from intelligence.feature_learning.representation_learning import RepresentationLearner
        self.assertIsNotNone(RepresentationLearner)


class TestAutoEncoder(unittest.TestCase):
    """Test AutoEncoder from representation_learning.py."""

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_instantiate(self):
        from intelligence.feature_learning.representation_learning import AutoEncoder
        ae = AutoEncoder(input_dim=20, latent_dim=5, hidden_dims=[64, 32])
        self.assertIsNotNone(ae)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_forward_pass(self):
        import torch
        from intelligence.feature_learning.representation_learning import AutoEncoder
        ae = AutoEncoder(input_dim=20, latent_dim=5, hidden_dims=[64, 32])
        x = torch.randn(8, 20)
        output = ae(x)
        self.assertIsNotNone(output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
