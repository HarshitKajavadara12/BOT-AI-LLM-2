"""
Proximal Policy Optimization (PPO) Agent
PPO algorithm for trading strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
from collections import deque


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.actor_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        shared_features = self.shared(state)
        
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_std)
        
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_mean, action_std, value = self.forward(state)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions."""
        action_mean, action_std, value = self.forward(state)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return log_prob, value.squeeze(-1), entropy


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, epochs: int = 10,
                 batch_size: int = 64, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }
        # Deterministic permutation counter for shuffling
        self._perm_counter = 0
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor)
        
        self.memory['states'].append(state)
        self.memory['actions'].append(action.numpy()[0])
        self.memory['log_probs'].append(log_prob.item())
        self.memory['values'].append(value.item())
        
        return action.numpy()[0]
    
    def store_transition(self, reward: float, done: bool):
        """Store transition in memory."""
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                advantage = 0
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            
            advantages.insert(0, advantage)
            next_value = values[t]
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values)
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO."""
        # Convert lists to arrays
        states = np.array(self.memory['states'])
        actions = np.array(self.memory['actions'])
        old_log_probs = np.array(self.memory['log_probs'])
        values = np.array(self.memory['values'])
        rewards = np.array(self.memory['rewards'])
        dones = np.array(self.memory['dones'])
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        
        # PPO update
        for _ in range(self.epochs):
            # Deterministic shuffle: cyclic roll of indices for reproducibility
            n = len(states)
            if n == 0:
                indices = np.array([])
            else:
                base = np.arange(n)
                shift = int(self._perm_counter % n)
                indices = np.roll(base, -shift)
                self._perm_counter += 1
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                   1 + self.clip_epsilon) * batch_advantages
                
                # Compute losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (batch_returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()
                
                loss = actor_loss + self.value_coef * critic_loss + \
                       self.entropy_coef * entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear memory
        for key in self.memory:
            self.memory[key] = []
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
