"""
Model-Based Reinforcement Learning
World models and planning for trading
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional
from collections import deque


class WorldModel(nn.Module):
    """Neural network world model."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(WorldModel, self).__init__()
        
        # Transition model
        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Reward model
        self.reward = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Done model
        self.done = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next state, reward, and done flag."""
        x = torch.cat([state, action], dim=-1)
        
        next_state = self.transition(x)
        reward = self.reward(x)
        done = self.done(x)
        
        return next_state, reward, done
    
    def predict_rollout(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict rollout for a sequence of actions."""
        states = []
        rewards = []
        
        current_state = state
        
        for t in range(actions.shape[1]):
            next_state, reward, _ = self.forward(current_state, actions[:, t, :])
            states.append(next_state)
            rewards.append(reward)
            current_state = next_state
        
        states = torch.stack(states, dim=1)
        rewards = torch.stack(rewards, dim=1)
        
        return states, rewards


class DynaQ:
    """Dyna-Q algorithm combining model-based and model-free RL."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 3e-4, gamma: float = 0.99,
                 planning_steps: int = 10):
        
        self.gamma = gamma
        self.planning_steps = planning_steps
        
        # World model
        self.world_model = WorldModel(state_dim, action_dim)
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=lr)
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.replay_buffer = deque(maxlen=100000)
        # Deterministic counters for reproducible sampling
        self._deterministic_counter = 0
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Epsilon-greedy action selection."""
        # Deterministic epsilon behavior: use neutral action when exploring
        if float(epsilon) > 0.5:
            return np.array([0.0])

        # Deterministic candidate actions and select best
        state_tensor = torch.FloatTensor(state).unsqueeze(0).repeat(100, 1)
        actions = torch.FloatTensor(np.linspace(-1.0, 1.0, 100).reshape(100, 1))
        
        with torch.no_grad():
            q_values = self.q_network(torch.cat([state_tensor, actions], dim=1))
        
        best_idx = q_values.argmax()
        return actions[best_idx].numpy()
    
    def update_world_model(self, state: np.ndarray, action: np.ndarray,
                          next_state: np.ndarray, reward: float, done: bool):
        """Update world model with real experience."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0)
        done_tensor = torch.FloatTensor([done]).unsqueeze(0)
        
        # Predict
        pred_next_state, pred_reward, pred_done = self.world_model(state_tensor, action_tensor)
        
        # Compute loss
        state_loss = nn.MSELoss()(pred_next_state, next_state_tensor)
        reward_loss = nn.MSELoss()(pred_reward, reward_tensor)
        done_loss = nn.BCELoss()(pred_done, done_tensor)
        
        loss = state_loss + reward_loss + done_loss
        
        # Update
        self.world_optimizer.zero_grad()
        loss.backward()
        self.world_optimizer.step()
        
        # Store experience
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def planning_update(self):
        """Perform planning updates using the world model."""
        if len(self.replay_buffer) < 32:
            return
        
        for _ in range(self.planning_steps):
            # Deterministic sampling from replay buffer (cyclic)
            idx = int(self._deterministic_counter % len(self.replay_buffer))
            state, _, _, _, _ = self.replay_buffer[idx]

            # Deterministic action (neutral)
            action = np.array([0.0])

            # Advance deterministic counter
            self._deterministic_counter += 1
            
            # Simulate with world model
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            
            with torch.no_grad():
                next_state, reward, done = self.world_model(state_tensor, action_tensor)
            
            # Update Q-function with simulated experience
            self._update_q(state_tensor, action_tensor, reward, next_state, done)
    
    def _update_q(self, state: torch.Tensor, action: torch.Tensor,
                  reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor):
        """Update Q-function."""
        # Current Q-value
        q_value = self.q_network(torch.cat([state, action], dim=1))
        
        # Target Q-value (using random action for next state - simplified)
        with torch.no_grad():
            # Deterministic next actions (zeros)
            next_actions = torch.zeros((state.shape[0], 1), dtype=torch.float32)
            next_q = self.q_network(torch.cat([next_state, next_actions], dim=1))
            target_q = reward + (1 - done) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(q_value, target_q)
        
        # Update
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()


class MBPO:
    """Model-Based Policy Optimization."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 rollout_length: int = 5, num_models: int = 5):
        
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.num_models = num_models
        
        # Ensemble of world models
        self.world_models = [WorldModel(state_dim, action_dim) for _ in range(num_models)]
        self.world_optimizers = [optim.Adam(model.parameters(), lr=lr) 
                                for model in self.world_models]
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Buffers
        self.real_buffer = deque(maxlen=100000)
        self.model_buffer = deque(maxlen=100000)
        # Deterministic model selection counter
        self._model_counter = 0
    
    def generate_model_rollouts(self, batch_size: int = 256):
        """Generate synthetic rollouts using world models."""
        if len(self.real_buffer) < batch_size:
            return
        
        # Sample real states
        # Deterministic spread of indices
        indices = np.linspace(0, len(self.real_buffer) - 1, batch_size, dtype=int)
        states = torch.FloatTensor([self.real_buffer[int(i)][0] for i in indices])
        
        # Generate rollouts
        for t in range(self.rollout_length):
            # Get actions from policy
            actions = self.policy(states)
            
            # Deterministic model selection (round-robin)
            model_idx = int(self._model_counter % self.num_models)
            model = self.world_models[model_idx]
            self._model_counter += 1
            
            # Predict next states and rewards
            with torch.no_grad():
                next_states, rewards, dones = model(states, actions)
            
            # Store in model buffer
            for i in range(batch_size):
                self.model_buffer.append((
                    states[i].numpy(),
                    actions[i].numpy(),
                    rewards[i].item(),
                    next_states[i].numpy(),
                    dones[i].item() > 0.5
                ))
            
            states = next_states
    
    def train_world_models(self, batch_size: int = 256):
        """Train ensemble of world models."""
        if len(self.real_buffer) < batch_size:
            return
        
        # Deterministic batch sampling
        indices = np.linspace(0, len(self.real_buffer) - 1, batch_size, dtype=int)
        batch = [self.real_buffer[int(i)] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Train each model in ensemble
        for model, optimizer in zip(self.world_models, self.world_optimizers):
            pred_next_states, pred_rewards, pred_dones = model(states, actions)
            
            state_loss = nn.MSELoss()(pred_next_states, next_states)
            reward_loss = nn.MSELoss()(pred_rewards, rewards)
            done_loss = nn.BCELoss()(pred_dones, dones)
            
            loss = state_loss + reward_loss + done_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
