"""
Test Suite for Machine Learning Models

This module contains comprehensive unit tests for machine learning models including
deep learning, reinforcement learning, meta-learning, probabilistic models, and
feature learning components.
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from intelligence.deep_learning.lstm_models import LSTMPredictor
    from intelligence.deep_learning.cnn_models import CNNFeatureExtractor
    from intelligence.deep_learning.transformer_models import TransformerPredictor
    from intelligence.deep_learning.attention_models import AttentionModel
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from intelligence.reinforcement_learning.dqn_agent import DQNAgent
from intelligence.reinforcement_learning.ppo_agent import PPOAgent
from intelligence.reinforcement_learning.trading_environment import TradingEnvironment
from intelligence.meta_learning.maml import ModelAgnosticMetaLearning
from intelligence.meta_learning.few_shot_learning import FewShotLearner
from intelligence.probabilistic_ml.gaussian_processes import GaussianProcess
from intelligence.probabilistic_ml.bayesian_networks import BayesianNetwork
from intelligence.probabilistic_ml.variational_inference import VariationalAutoencoder
from intelligence.feature_learning.autoencoder import FeatureAutoencoder
from intelligence.feature_learning.representation_learning import RepresentationLearner


class TestLSTMModels(unittest.TestCase):
    """Test cases for LSTM models."""
    
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        np.random.seed(42)
        self.sequence_length = 10
        self.n_features = 5
        self.n_samples = 100
        
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def test_lstm_model_creation(self):
        """Test LSTM model architecture creation."""
        model = LSTMPredictor(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            hidden_units=64,
            n_layers=2
        )
        
        self.assertIsNotNone(model.model)
        
        # Check input shape
        input_shape = model.model.input_shape
        self.assertEqual(input_shape[1], self.sequence_length)
        self.assertEqual(input_shape[2], self.n_features)
        
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def test_lstm_training(self):
        """Test LSTM model training."""
        # Generate synthetic data
        X = np.random.randn(self.n_samples, self.sequence_length, self.n_features)
        y = np.random.randn(self.n_samples, 1)
        
        model = LSTMPredictor(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            hidden_units=32
        )
        
        history = model.fit(X, y, epochs=5, batch_size=16, verbose=0)
        
        # Check that loss decreased
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        self.assertLess(final_loss, initial_loss)
        
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def test_lstm_prediction(self):
        """Test LSTM model prediction."""
        X = np.random.randn(self.n_samples, self.sequence_length, self.n_features)
        y = np.random.randn(self.n_samples, 1)
        
        model = LSTMPredictor(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            hidden_units=32
        )
        
        model.fit(X, y, epochs=3, batch_size=16, verbose=0)
        
        # Make predictions
        X_test = np.random.randn(10, self.sequence_length, self.n_features)
        predictions = model.predict(X_test)
        
        self.assertEqual(predictions.shape[0], 10)
        
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def test_bidirectional_lstm(self):
        """Test bidirectional LSTM."""
        model = LSTMPredictor(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            hidden_units=32,
            bidirectional=True
        )
        
        X = np.random.randn(20, self.sequence_length, self.n_features)
        predictions = model.predict(X)
        
        self.assertEqual(predictions.shape[0], 20)


class TestCNNModels(unittest.TestCase):
    """Test cases for CNN models."""
    
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        np.random.seed(42)
        
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def test_cnn_feature_extraction(self):
        """Test CNN for feature extraction from time series."""
        window_size = 50
        n_features = 10
        
        model = CNNFeatureExtractor(
            window_size=window_size,
            n_features=n_features,
            n_filters=[32, 64],
            kernel_sizes=[3, 3]
        )
        
        # Generate data
        X = np.random.randn(100, window_size, n_features)
        features = model.extract_features(X)
        
        self.assertEqual(features.shape[0], 100)
        self.assertGreater(features.shape[1], 0)
        
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def test_cnn_pattern_recognition(self):
        """Test CNN for pattern recognition in financial data."""
        # Create simple patterns
        n_samples = 100
        window_size = 20
        
        # Pattern: uptrend
        uptrend = np.linspace(0, 1, window_size).reshape(1, -1, 1)
        # Pattern: downtrend
        downtrend = np.linspace(1, 0, window_size).reshape(1, -1, 1)
        
        X = np.vstack([
            uptrend + np.random.randn(n_samples//2, window_size, 1) * 0.1,
            downtrend + np.random.randn(n_samples//2, window_size, 1) * 0.1
        ])
        y = np.array([1] * (n_samples//2) + [0] * (n_samples//2))
        
        model = CNNFeatureExtractor(
            window_size=window_size,
            n_features=1,
            n_filters=[16, 32],
            kernel_sizes=[3, 3]
        )
        
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)
        
        # Test prediction
        test_up = uptrend + np.random.randn(1, window_size, 1) * 0.1
        test_down = downtrend + np.random.randn(1, window_size, 1) * 0.1
        
        pred_up = model.predict(test_up)
        pred_down = model.predict(test_down)
        
        # Should classify correctly (with some tolerance)
        self.assertGreater(pred_up[0], 0.3)  # Should predict uptrend
        self.assertLess(pred_down[0], 0.7)  # Should predict downtrend


class TestTransformerModels(unittest.TestCase):
    """Test cases for Transformer models."""
    
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def test_transformer_architecture(self):
        """Test Transformer model creation."""
        model = TransformerPredictor(
            sequence_length=20,
            n_features=10,
            d_model=64,
            n_heads=4,
            n_layers=2
        )
        
        self.assertIsNotNone(model.model)
        
    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def test_attention_mechanism(self):
        """Test attention mechanism."""
        attention = AttentionModel(d_model=64, n_heads=4)
        
        # Random input
        x = tf.random.normal((2, 10, 64))  # (batch, seq_len, d_model)
        output, attention_weights = attention(x, return_attention_weights=True)
        
        self.assertEqual(output.shape, x.shape)
        self.assertIsNotNone(attention_weights)


class TestReinforcementLearning(unittest.TestCase):
    """Test cases for reinforcement learning agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_trading_environment(self):
        """Test trading environment setup."""
        # Generate price data
        prices = 100 + np.random.randn(1000).cumsum()
        
        env = TradingEnvironment(
            prices=prices,
            initial_capital=10000,
            transaction_cost=0.001
        )
        
        state = env.reset()
        self.assertIsNotNone(state)
        
        # Test step
        action = 1  # Buy
        next_state, reward, done, info = env.step(action)
        
        self.assertIsNotNone(next_state)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        
    def test_dqn_agent_creation(self):
        """Test DQN agent initialization."""
        state_size = 10
        action_size = 3
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_layers=[64, 64]
        )
        
        self.assertEqual(agent.state_size, state_size)
        self.assertEqual(agent.action_size, action_size)
        
    def test_dqn_action_selection(self):
        """Test DQN action selection."""
        agent = DQNAgent(state_size=5, action_size=3)
        
        state = np.random.randn(5)
        action = agent.act(state, epsilon=0.1)
        
        self.assertIn(action, [0, 1, 2])
        
    def test_dqn_training_step(self):
        """Test DQN training step."""
        agent = DQNAgent(state_size=5, action_size=3, learning_rate=0.001)
        
        # Add experience
        for _ in range(100):
            state = np.random.randn(5)
            action = np.random.randint(3)
            reward = np.random.randn()
            next_state = np.random.randn(5)
            done = False
            
            agent.remember(state, action, reward, next_state, done)
        
        # Train
        initial_loss = agent.replay(batch_size=32)
        
        # Train more
        for _ in range(50):
            agent.remember(
                np.random.randn(5),
                np.random.randint(3),
                np.random.randn(),
                np.random.randn(5),
                False
            )
        
        final_loss = agent.replay(batch_size=32)
        
        # Both should return valid loss
        self.assertIsInstance(initial_loss, (float, type(None)))
        
    def test_ppo_agent(self):
        """Test PPO agent."""
        agent = PPOAgent(
            state_size=10,
            action_size=3,
            hidden_layers=[64, 64]
        )
        
        state = np.random.randn(10)
        action, log_prob = agent.select_action(state)
        
        self.assertIn(action, [0, 1, 2])
        self.assertIsNotNone(log_prob)
        
    def test_trading_episode(self):
        """Test complete trading episode."""
        prices = 100 + np.random.randn(100).cumsum()
        env = TradingEnvironment(prices=prices, initial_capital=10000)
        agent = DQNAgent(state_size=env.state_size, action_size=3)
        
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.act(state, epsilon=0.5)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
        
        # Should complete episode
        self.assertGreater(steps, 0)


class TestMetaLearning(unittest.TestCase):
    """Test cases for meta-learning algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_maml_initialization(self):
        """Test MAML model initialization."""
        maml = ModelAgnosticMetaLearning(
            input_dim=10,
            output_dim=1,
            hidden_layers=[32, 32],
            inner_lr=0.01,
            outer_lr=0.001
        )
        
        self.assertIsNotNone(maml.model)
        
    def test_maml_adaptation(self):
        """Test MAML fast adaptation."""
        maml = ModelAgnosticMetaLearning(
            input_dim=5,
            output_dim=1,
            hidden_layers=[16],
            inner_lr=0.1
        )
        
        # Support set
        X_support = np.random.randn(10, 5)
        y_support = np.random.randn(10, 1)
        
        # Adapt to task
        adapted_model = maml.adapt(X_support, y_support, steps=5)
        
        self.assertIsNotNone(adapted_model)
        
    def test_few_shot_learning(self):
        """Test few-shot learning."""
        learner = FewShotLearner(
            input_dim=10,
            n_way=3,
            k_shot=5
        )
        
        # Create few-shot task
        support_set = {
            'X': np.random.randn(15, 10),  # 3 classes * 5 shots
            'y': np.array([0]*5 + [1]*5 + [2]*5)
        }
        
        query_set = {
            'X': np.random.randn(9, 10),  # 3 per class
            'y': np.array([0]*3 + [1]*3 + [2]*3)
        }
        
        # Train on support set
        learner.train(support_set)
        
        # Evaluate on query set
        accuracy = learner.evaluate(query_set)
        
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        
    def test_meta_training_episode(self):
        """Test meta-training episode."""
        maml = ModelAgnosticMetaLearning(
            input_dim=5,
            output_dim=1,
            hidden_layers=[16],
            inner_lr=0.1,
            outer_lr=0.01
        )
        
        # Generate tasks
        tasks = []
        for _ in range(5):
            X = np.random.randn(20, 5)
            y = np.random.randn(20, 1)
            tasks.append((X, y))
        
        # Meta-training step
        loss = maml.meta_train_step(tasks, k_support=10)
        
        self.assertIsNotNone(loss)


class TestProbabilisticML(unittest.TestCase):
    """Test cases for probabilistic ML models."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_gaussian_process_regression(self):
        """Test Gaussian Process regression."""
        gp = GaussianProcess(kernel='rbf', length_scale=1.0, noise_level=0.1)
        
        # Training data
        X_train = np.linspace(0, 10, 20).reshape(-1, 1)
        y_train = np.sin(X_train).ravel() + np.random.randn(20) * 0.1
        
        gp.fit(X_train, y_train)
        
        # Prediction
        X_test = np.linspace(0, 10, 50).reshape(-1, 1)
        mean, std = gp.predict(X_test, return_std=True)
        
        self.assertEqual(len(mean), 50)
        self.assertEqual(len(std), 50)
        self.assertTrue(np.all(std > 0))
        
    def test_gp_confidence_intervals(self):
        """Test GP confidence intervals."""
        gp = GaussianProcess(kernel='rbf')
        
        X_train = np.array([[0], [1], [2], [3], [4]])
        y_train = np.array([0, 1, 2, 3, 4])
        
        gp.fit(X_train, y_train)
        
        X_test = np.array([[2.5]])
        mean, std = gp.predict(X_test, return_std=True)
        
        # Prediction should be close to 2.5
        self.assertGreater(mean[0], 2.0)
        self.assertLess(mean[0], 3.0)
        
    def test_bayesian_network(self):
        """Test Bayesian Network."""
        bn = BayesianNetwork()
        
        # Add nodes
        bn.add_node('Market', values=['Bull', 'Bear'])
        bn.add_node('Sector', values=['DeFi', 'L1'])
        bn.add_node('Crypto', values=['Up', 'Down'])
        
        # Add edges
        bn.add_edge('Market', 'Crypto')
        bn.add_edge('Sector', 'Crypto')
        
        # Learn parameters from data
        data = pd.DataFrame({
            'Market': ['Bull', 'Bull', 'Bear', 'Bear'] * 25,
            'Sector': ['DeFi', 'L1', 'DeFi', 'L1'] * 25,
            'Crypto': ['Up', 'Up', 'Down', 'Down'] * 25
        })
        
        bn.fit(data)
        
        # Inference
        prob = bn.predict_proba(
            evidence={'Market': 'Bull', 'Sector': 'DeFi'},
            query='Crypto'
        )
        
        self.assertIn('Up', prob)
        self.assertIn('Down', prob)
        
    def test_variational_autoencoder(self):
        """Test Variational Autoencoder."""
        latent_dim = 10
        input_dim = 50
        
        vae = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_layers=[32, 16]
        )
        
        # Generate data
        X = np.random.randn(100, input_dim)
        
        # Train
        history = vae.fit(X, epochs=5, batch_size=16, verbose=0)
        
        # Encode
        z_mean, z_log_var = vae.encode(X[:10])
        
        self.assertEqual(z_mean.shape[1], latent_dim)
        self.assertEqual(z_log_var.shape[1], latent_dim)
        
        # Decode
        reconstructed = vae.decode(z_mean)
        self.assertEqual(reconstructed.shape[1], input_dim)
        
    def test_vae_sampling(self):
        """Test VAE sampling from latent space."""
        vae = VariationalAutoencoder(
            input_dim=20,
            latent_dim=5,
            hidden_layers=[16]
        )
        
        X = np.random.randn(50, 20)
        vae.fit(X, epochs=3, batch_size=16, verbose=0)
        
        # Sample from latent space
        samples = vae.sample(n_samples=10)
        
        self.assertEqual(samples.shape[0], 10)
        self.assertEqual(samples.shape[1], 20)


class TestFeatureLearning(unittest.TestCase):
    """Test cases for feature learning."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_autoencoder_dimensionality_reduction(self):
        """Test autoencoder for dimensionality reduction."""
        input_dim = 50
        encoding_dim = 10
        
        autoencoder = FeatureAutoencoder(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            hidden_layers=[30, 20]
        )
        
        # Generate data
        X = np.random.randn(100, input_dim)
        
        # Train
        history = autoencoder.fit(X, epochs=10, batch_size=16, verbose=0)
        
        # Encode
        encoded = autoencoder.encode(X)
        
        self.assertEqual(encoded.shape[1], encoding_dim)
        
        # Decode
        decoded = autoencoder.decode(encoded)
        
        self.assertEqual(decoded.shape[1], input_dim)
        
        # Reconstruction error should decrease
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        self.assertLess(final_loss, initial_loss)
        
    def test_sparse_autoencoder(self):
        """Test sparse autoencoder."""
        autoencoder = FeatureAutoencoder(
            input_dim=30,
            encoding_dim=15,
            hidden_layers=[20],
            sparsity_constraint=True,
            sparsity_weight=0.01
        )
        
        X = np.random.randn(80, 30)
        autoencoder.fit(X, epochs=5, batch_size=16, verbose=0)
        
        encoded = autoencoder.encode(X)
        
        # Sparse encoding should have many small values
        sparsity = np.mean(np.abs(encoded) < 0.1)
        self.assertGreater(sparsity, 0.1)
        
    def test_representation_learning(self):
        """Test representation learning."""
        learner = RepresentationLearner(
            input_dim=40,
            representation_dim=10,
            learning_method='contrastive'
        )
        
        X = np.random.randn(100, 40)
        
        # Learn representations
        representations = learner.fit_transform(X, epochs=5, batch_size=16)
        
        self.assertEqual(representations.shape[0], 100)
        self.assertEqual(representations.shape[1], 10)
        
    def test_contrastive_learning(self):
        """Test contrastive learning."""
        learner = RepresentationLearner(
            input_dim=20,
            representation_dim=5,
            learning_method='contrastive',
            temperature=0.5
        )
        
        # Create positive pairs (augmented versions)
        X = np.random.randn(50, 20)
        X_augmented = X + np.random.randn(50, 20) * 0.1
        
        # Train
        learner.fit_contrastive(X, X_augmented, epochs=5, batch_size=16)
        
        # Representations
        repr1 = learner.transform(X[:5])
        repr2 = learner.transform(X_augmented[:5])
        
        # Positive pairs should be close
        distances = np.linalg.norm(repr1 - repr2, axis=1)
        mean_distance = np.mean(distances)
        
        # Should be reasonably close
        self.assertLess(mean_distance, 2.0)


class TestModelEnsembles(unittest.TestCase):
    """Test cases for model ensembles."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_ensemble_prediction(self):
        """Test ensemble prediction combination."""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        
        # Generate data
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        # Train multiple models
        models = [
            RandomForestRegressor(n_estimators=10, random_state=42),
            GradientBoostingRegressor(n_estimators=10, random_state=42),
            Ridge()
        ]
        
        for model in models:
            model.fit(X, y)
        
        # Ensemble prediction
        X_test = np.random.randn(20, 10)
        predictions = np.array([model.predict(X_test) for model in models])
        
        # Average
        ensemble_pred = np.mean(predictions, axis=0)
        
        self.assertEqual(len(ensemble_pred), 20)
        
    def test_weighted_ensemble(self):
        """Test weighted ensemble."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20)
        
        models = [
            RandomForestRegressor(n_estimators=10, random_state=42),
            Ridge()
        ]
        
        # Train models and get validation predictions
        val_preds = []
        for model in models:
            model.fit(X_train, y_train)
            val_preds.append(model.predict(X_val))
        
        # Calculate weights based on validation performance
        from sklearn.metrics import mean_squared_error
        errors = [mean_squared_error(y_val, pred) for pred in val_preds]
        weights = 1 / (np.array(errors) + 1e-6)
        weights = weights / weights.sum()
        
        # Weighted prediction
        X_test = np.random.randn(10, 5)
        test_preds = np.array([model.predict(X_test) for model in models])
        weighted_pred = np.average(test_preds, axis=0, weights=weights)
        
        self.assertEqual(len(weighted_pred), 10)


def run_tests():
    """Run all model tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    if TF_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestLSTMModels))
        suite.addTests(loader.loadTestsFromTestCase(TestCNNModels))
        suite.addTests(loader.loadTestsFromTestCase(TestTransformerModels))
    
    suite.addTests(loader.loadTestsFromTestCase(TestReinforcementLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestMetaLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestProbabilisticML))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestModelEnsembles))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
