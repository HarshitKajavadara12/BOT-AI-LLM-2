"""
Reinforcement Learning Agents for QUANTUM-FORGE
Implements advanced RL algorithms including DQN, PPO, SAC, and multi-agent systems for trading.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import warnings
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time
from collections import deque, namedtuple
import random
import pickle
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AgentType(Enum):
    """Types of reinforcement learning agents."""
    DQN = "deep_q_network"
    DDQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"
    PPO = "proximal_policy_optimization"
    SAC = "soft_actor_critic"
    TD3 = "twin_delayed_ddpg"
    A3C = "asynchronous_advantage_actor_critic"
    RAINBOW = "rainbow_dqn"

class ActionSpace(Enum):
    """Types of action spaces."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_DISCRETE = "multi_discrete"

@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    agent_type: AgentType
    state_dim: int
    action_dim: int
    action_space: ActionSpace
    hidden_dim: int
    learning_rate: float
    gamma: float  # Discount factor
    tau: float   # Soft update parameter
    epsilon: float  # Exploration parameter
    epsilon_decay: float
    epsilon_min: float
    buffer_size: int
    batch_size: int
    update_frequency: int
    target_update_frequency: int

@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict = None

@dataclass
class TrainingMetrics:
    """Training metrics for RL agents."""
    episode: int
    total_reward: float
    episode_length: int
    loss: float
    epsilon: float
    q_value: float
    policy_entropy: Optional[float] = None

class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int):
        """Initialize replay buffer."""
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple('Experience', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state: np.ndarray, action: Union[int, np.ndarray], 
             reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences."""
        # Deterministic sampling: evenly spaced indices across the buffer for reproducibility
        n = len(self.buffer)
        if batch_size >= n:
            batch = list(self.buffer)
        else:
            indices = np.linspace(0, n - 1, batch_size, dtype=int)
            batch = [self.buffer[int(i)] for i in indices]
        
        states = torch.FloatTensor([e.state for e in batch]).to(DEVICE)
        actions = torch.LongTensor([e.action for e in batch]).to(DEVICE)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(DEVICE)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(DEVICE)
        dones = torch.BoolTensor([e.done for e in batch]).to(DEVICE)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """Initialize prioritized replay buffer."""
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: Union[int, np.ndarray],
             reward: float, next_state: np.ndarray, done: bool):
        """Add experience with maximum priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        experience = (state, action, reward, next_state, done)
        self.buffer[self.position] = experience
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample batch with importance weights."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        
        # Calculate sampling probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Deterministic systematic sampling based on cumulative probabilities
        cdf = np.cumsum(probs)
        # Evenly spaced thresholds in (0,1]
        thresholds = (np.arange(batch_size) + 0.5) / float(batch_size)
        indices = np.searchsorted(cdf, thresholds)
        
        # Sample experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Convert to tensors
        states = torch.FloatTensor([e[0] for e in experiences]).to(DEVICE)
        actions = torch.LongTensor([e[1] for e in experiences]).to(DEVICE)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(DEVICE)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(DEVICE)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(DEVICE)
        weights = torch.FloatTensor(weights).to(DEVICE)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class DQNNetwork(nn.Module):
    """Deep Q-Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize DQN."""
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN Network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize Dueling DQN."""
        super(DuelingDQNNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class ActorNetwork(nn.Module):
    """Actor network for policy-based methods."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_space: ActionSpace = ActionSpace.CONTINUOUS):
        """Initialize actor network."""
        super(ActorNetwork, self).__init__()
        
        self.action_space = action_space
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        if action_space == ActionSpace.DISCRETE:
            self.output = nn.Linear(hidden_dim, action_dim)
        else:  # Continuous
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        if self.action_space == ActionSpace.DISCRETE:
            action_probs = F.softmax(self.output(x), dim=-1)
            return action_probs
        else:
            mean = self.mean(x)
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, min=-20, max=2)
            return mean, log_std

class CriticNetwork(nn.Module):
    """Critic network for actor-critic methods."""
    
    def __init__(self, state_dim: int, action_dim: int = None, hidden_dim: int = 256):
        """Initialize critic network."""
        super(CriticNetwork, self).__init__()
        
        input_dim = state_dim + (action_dim if action_dim else 0)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state, action=None):
        """Forward pass."""
        if action is not None:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
            
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value

class DQNAgent:
    """Deep Q-Network agent."""
    
    def __init__(self, config: AgentConfig):
        """Initialize DQN agent."""
        self.config = config
        self.device = DEVICE
        
        # Networks
        if config.agent_type == AgentType.DUELING_DQN:
            self.q_network = DuelingDQNNetwork(
                config.state_dim, config.action_dim, config.hidden_dim
            ).to(self.device)
            self.target_network = DuelingDQNNetwork(
                config.state_dim, config.action_dim, config.hidden_dim
            ).to(self.device)
        else:
            self.q_network = DQNNetwork(
                config.state_dim, config.action_dim, config.hidden_dim
            ).to(self.device)
            self.target_network = DQNNetwork(
                config.state_dim, config.action_dim, config.hidden_dim
            ).to(self.device)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        if config.agent_type == AgentType.RAINBOW:
            self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training parameters
        self.epsilon = config.epsilon
        self.step_count = 0
        self.training_metrics = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        
        # Sample batch
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.config.batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.config.batch_size)
            weights = None
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.config.agent_type == AgentType.DDQN:
                # Double DQN
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + \
                             (self.config.gamma * next_q_values * (~dones).unsqueeze(1))
        
        # Compute loss
        if weights is not None:
            # Prioritized replay
            td_errors = current_q_values - target_q_values
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
            
            # Update priorities
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, priorities + 1e-6)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.config.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
        
        return loss.item()

class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, config: AgentConfig):
        """Initialize PPO agent."""
        self.config = config
        self.device = DEVICE
        
        # Networks
        self.actor = ActorNetwork(
            config.state_dim, config.action_dim, config.hidden_dim, config.action_space
        ).to(self.device)
        
        self.critic = CriticNetwork(
            config.state_dim, hidden_dim=config.hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.ppo_epochs = 10
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.training_metrics = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_tensor)
            
            if self.config.action_space == ActionSpace.DISCRETE:
                action_probs = self.actor(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                return action.item(), log_prob.item(), value.item()
            else:
                mean, log_std = self.actor(state_tensor)
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                return action.cpu().numpy()[0], log_prob.item(), value.item()
    
    def store_experience(self, state: np.ndarray, action: Union[int, np.ndarray],
                        reward: float, log_prob: float, value: float, done: bool):
        """Store experience for batch training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, next_value: float = 0.0) -> Tuple:
        """Compute returns and advantages using GAE."""
        returns = []
        advantages = []
        
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value_i = next_value
            else:
                next_value_i = self.values[i + 1]
            
            delta = self.rewards[i] + self.config.gamma * next_value_i * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.config.gamma * self.config.tau * (1 - self.dones[i]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        return returns, advantages
    
    def train_step(self, next_value: float = 0.0) -> Dict:
        """Perform PPO training update."""
        if len(self.states) == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if self.config.action_space == ActionSpace.DISCRETE:
            actions = torch.LongTensor(self.actions).to(self.device)
        else:
            actions = torch.FloatTensor(self.actions).to(self.device)
        
        # PPO training loop
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(self.ppo_epochs):
            # Actor loss
            if self.config.action_space == ActionSpace.DISCRETE:
                action_probs = self.actor(states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            else:
                mean, log_std = self.actor(states)
                std = log_std.exp()
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            
            # Critic loss
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
        
        # Clear storage
        self.clear_storage()
        
        return {
            'actor_loss': total_actor_loss / self.ppo_epochs,
            'critic_loss': total_critic_loss / self.ppo_epochs,
            'entropy': total_entropy / self.ppo_epochs
        }
    
    def clear_storage(self):
        """Clear experience storage."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

class SACAgent:
    """Soft Actor-Critic agent for continuous action spaces."""
    
    def __init__(self, config: AgentConfig):
        """Initialize SAC agent."""
        self.config = config
        self.device = DEVICE
        
        # Networks
        self.actor = ActorNetwork(
            config.state_dim, config.action_dim, config.hidden_dim, ActionSpace.CONTINUOUS
        ).to(self.device)
        
        self.critic1 = CriticNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        self.critic2 = CriticNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        # Target critics
        self.target_critic1 = CriticNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        self.target_critic2 = CriticNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.learning_rate)
        
        # Automatic entropy tuning
        self.target_entropy = -config.action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        self.training_metrics = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            
            if training:
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
            else:
                action = mean
            
            action = torch.tanh(action)  # Bound action to [-1, 1]
            
        return action.cpu().numpy()[0]
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Dict:
        """Perform SAC training update."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha_loss': 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.config.batch_size)
        
        # Convert actions to float for continuous control
        actions = actions.float()
        
        # Update critics
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_dist = Normal(next_mean, next_std)
            next_actions = next_dist.sample()
            next_actions = torch.tanh(next_actions)
            
            next_log_probs = next_dist.log_prob(next_actions).sum(dim=-1, keepdim=True)
            next_log_probs -= torch.log(1 - next_actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            alpha = self.log_alpha.exp()
            target_q = rewards.unsqueeze(1) + \
                      (self.config.gamma * (~dones).unsqueeze(1) * (next_q - alpha * next_log_probs))
        
        # Critic loss
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor loss
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        actions_new = dist.rsample()  # Reparameterization trick
        actions_new = torch.tanh(actions_new)
        
        log_probs = dist.log_prob(actions_new).sum(dim=-1, keepdim=True)
        log_probs -= torch.log(1 - actions_new.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        q1_new = self.critic1(states, actions_new)
        q2_new = self.critic2(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Alpha loss (entropy regularization)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }
    
    def soft_update(self, local_model: nn.Module, target_model: nn.Module):
        """Soft update target network parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data
            )

class RLTrainingEnvironment:
    """RL training environment wrapper."""
    
    def __init__(self):
        """Initialize RL training environment."""
        self.agents = {}
        self.training_history = {}
    
    def register_agent(self, agent_name: str, agent: Union[DQNAgent, PPOAgent, SACAgent]):
        """Register RL agent."""
        self.agents[agent_name] = agent
        self.training_history[agent_name] = []
    
    def train_episode(self, agent_name: str, env_step_function: Callable,
                     max_steps: int = 1000) -> TrainingMetrics:
        """Train agent for one episode."""
        
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not registered")
        
        agent = self.agents[agent_name]
        episode_reward = 0.0
        episode_steps = 0
        episode_loss = 0.0
        
        # Initialize environment
        state = env_step_function('reset')
        
        for step in range(max_steps):
            # Select action
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, training=True)
                
                # Environment step
                next_state, reward, done, info = env_step_function('step', action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train
                if step % agent.config.update_frequency == 0:
                    loss = agent.train_step()
                    episode_loss += loss
                
                # Get current Q-value for metrics
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_value = agent.q_network(state_tensor).max().item()
                
            elif isinstance(agent, PPOAgent):
                action, log_prob, value = agent.select_action(state, training=True)
                
                # Environment step
                next_state, reward, done, info = env_step_function('step', action)
                
                # Store experience
                agent.store_experience(state, action, reward, log_prob, value, done)
                
                # Train at episode end or when buffer is full
                if done or len(agent.states) >= agent.config.batch_size:
                    next_value = 0.0
                    if not done:
                        _, _, next_value = agent.select_action(next_state, training=False)
                    
                    train_results = agent.train_step(next_value)
                    episode_loss += train_results['actor_loss']
                
                q_value = value  # Use value as proxy for Q-value
                
            elif isinstance(agent, SACAgent):
                action = agent.select_action(state, training=True)
                
                # Environment step
                next_state, reward, done, info = env_step_function('step', action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train
                if step % agent.config.update_frequency == 0:
                    train_results = agent.train_step()
                    episode_loss += train_results['actor_loss']
                
                # Get current Q-value for metrics
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_value = agent.critic1(state_tensor, action_tensor).item()
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # Create training metrics
        epsilon = getattr(agent, 'epsilon', 0.0)
        metrics = TrainingMetrics(
            episode=len(self.training_history[agent_name]),
            total_reward=episode_reward,
            episode_length=episode_steps,
            loss=episode_loss / max(episode_steps, 1),
            epsilon=epsilon,
            q_value=q_value
        )
        
        self.training_history[agent_name].append(metrics)
        return metrics
    
    def get_training_summary(self, agent_name: str) -> Dict:
        """Get training summary for agent."""
        
        if agent_name not in self.training_history:
            return {}
        
        history = self.training_history[agent_name]
        
        if not history:
            return {}
        
        recent_rewards = [m.total_reward for m in history[-100:]]
        recent_losses = [m.loss for m in history[-100:]]
        
        return {
            'total_episodes': len(history),
            'average_reward_recent': np.mean(recent_rewards),
            'best_reward': max(m.total_reward for m in history),
            'average_loss_recent': np.mean(recent_losses),
            'average_episode_length': np.mean([m.episode_length for m in history[-100:]]),
            'current_epsilon': history[-1].epsilon if hasattr(history[-1], 'epsilon') else 0.0
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Reinforcement Learning Agents...")
    
    # Simple trading environment simulator
    class SimpleTradingEnv:
        def __init__(self):
            self.reset()
            
        def reset(self):
            self.position = 0
            self.balance = 10000
            self.price = 100.0
            self.step_count = 0
            self.max_steps = 1000
            
            # Generate price series
            np.random.seed(42)
            self.prices = [100.0]
            for _ in range(self.max_steps):
                change = np.random.normal(0, 0.02)  # 2% volatility
                new_price = self.prices[-1] * (1 + change)
                self.prices.append(max(new_price, 0.1))  # Prevent negative prices
            
            return self.get_state()
        
        def get_state(self):
            # State: [current_price, position, balance_ratio, recent_returns]
            recent_prices = self.prices[max(0, self.step_count-4):self.step_count+1]
            if len(recent_prices) > 1:
                recent_returns = [(recent_prices[i] / recent_prices[i-1] - 1) 
                                for i in range(1, len(recent_prices))]
            else:
                recent_returns = [0.0] * 4
            
            # Pad returns to fixed length
            while len(recent_returns) < 4:
                recent_returns.insert(0, 0.0)
            
            state = [
                self.price / 100.0,  # Normalized price
                self.position / 100.0,  # Normalized position
                self.balance / 10000.0,  # Normalized balance
            ] + recent_returns
            
            return np.array(state, dtype=np.float32)
        
        def step(self, action):
            if self.step_count >= self.max_steps:
                return self.get_state(), 0.0, True, {}
            
            old_portfolio_value = self.balance + self.position * self.price
            
            # Actions: 0=hold, 1=buy, 2=sell
            if action == 1 and self.balance >= self.price:  # Buy
                shares_to_buy = int(self.balance // self.price)
                cost = shares_to_buy * self.price
                self.position += shares_to_buy
                self.balance -= cost
                
            elif action == 2 and self.position > 0:  # Sell
                revenue = self.position * self.price
                self.balance += revenue
                self.position = 0
            
            # Move to next time step
            self.step_count += 1
            if self.step_count < len(self.prices):
                self.price = self.prices[self.step_count]
            
            # Calculate reward (portfolio value change)
            new_portfolio_value = self.balance + self.position * self.price
            reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
            
            # Episode done
            done = self.step_count >= self.max_steps
            
            return self.get_state(), reward, done, {'portfolio_value': new_portfolio_value}
    
    # Environment step function for RL training
    trading_env = SimpleTradingEnv()
    
    def env_step_function(action_type, action=None):
        if action_type == 'reset':
            return trading_env.reset()
        elif action_type == 'step':
            return trading_env.step(action)
    
    # Test DQN Agent
    print("\n=== Testing DQN Agent ===")
    
    dqn_config = AgentConfig(
        agent_type=AgentType.DQN,
        state_dim=7,  # State dimension from our environment
        action_dim=3,  # 3 actions: hold, buy, sell
        action_space=ActionSpace.DISCRETE,
        hidden_dim=128,
        learning_rate=0.001,
        gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=32,
        update_frequency=4,
        target_update_frequency=100
    )
    
    # Initialize training environment
    training_env = RLTrainingEnvironment()
    
    # Create and register DQN agent
    dqn_agent = DQNAgent(dqn_config)
    training_env.register_agent("dqn_trader", dqn_agent)
    
    # Training loop
    print("Training DQN agent...")
    for episode in range(50):  # Reduced for testing
        metrics = training_env.train_episode("dqn_trader", env_step_function, max_steps=200)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={metrics.total_reward:.4f}, "
                  f"Steps={metrics.episode_length}, Loss={metrics.loss:.6f}, "
                  f"Epsilon={metrics.epsilon:.3f}")
    
    dqn_summary = training_env.get_training_summary("dqn_trader")
    print(f"DQN Training Summary:")
    print(f"  Total episodes: {dqn_summary['total_episodes']}")
    print(f"  Average recent reward: {dqn_summary['average_reward_recent']:.4f}")
    print(f"  Best reward: {dqn_summary['best_reward']:.4f}")
    print(f"  Final epsilon: {dqn_summary['current_epsilon']:.3f}")
    
    # Test PPO Agent
    print("\n=== Testing PPO Agent ===")
    
    ppo_config = AgentConfig(
        agent_type=AgentType.PPO,
        state_dim=7,
        action_dim=3,
        action_space=ActionSpace.DISCRETE,
        hidden_dim=128,
        learning_rate=0.0003,
        gamma=0.99,
        tau=0.95,  # GAE lambda
        epsilon=0.2,  # PPO clip parameter
        epsilon_decay=1.0,  # No decay for PPO
        epsilon_min=0.2,
        buffer_size=2048,  # PPO batch size
        batch_size=64,
        update_frequency=2048,  # Update after collecting batch
        target_update_frequency=1
    )
    
    ppo_agent = PPOAgent(ppo_config)
    training_env.register_agent("ppo_trader", ppo_agent)
    
    print("Training PPO agent...")
    for episode in range(30):  # Reduced for testing
        metrics = training_env.train_episode("ppo_trader", env_step_function, max_steps=200)
        
        if episode % 5 == 0:
            print(f"Episode {episode}: Reward={metrics.total_reward:.4f}, "
                  f"Steps={metrics.episode_length}, Loss={metrics.loss:.6f}")
    
    ppo_summary = training_env.get_training_summary("ppo_trader")
    print(f"PPO Training Summary:")
    print(f"  Total episodes: {ppo_summary['total_episodes']}")
    print(f"  Average recent reward: {ppo_summary['average_reward_recent']:.4f}")
    print(f"  Best reward: {ppo_summary['best_reward']:.4f}")
    
    # Test SAC Agent with continuous actions
    print("\n=== Testing SAC Agent ===")
    
    # Modified environment for continuous actions
    class ContinuousTradingEnv(SimpleTradingEnv):
        def step(self, action):
            if self.step_count >= self.max_steps:
                return self.get_state(), 0.0, True, {}
            
            old_portfolio_value = self.balance + self.position * self.price
            
            # Continuous action: -1 to 1 (sell all to buy all)
            action = np.clip(action[0], -1, 1)
            
            target_position_ratio = (action + 1) / 2  # Convert to 0-1
            max_position = (self.balance + self.position * self.price) / self.price
            target_position = target_position_ratio * max_position
            
            position_change = target_position - self.position
            
            if position_change > 0 and self.balance >= position_change * self.price:
                # Buy
                cost = position_change * self.price
                self.position += position_change
                self.balance -= cost
            elif position_change < 0 and self.position >= -position_change:
                # Sell
                revenue = -position_change * self.price
                self.position += position_change
                self.balance += revenue
            
            # Move to next time step
            self.step_count += 1
            if self.step_count < len(self.prices):
                self.price = self.prices[self.step_count]
            
            # Calculate reward
            new_portfolio_value = self.balance + self.position * self.price
            reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
            
            done = self.step_count >= self.max_steps
            
            return self.get_state(), reward, done, {'portfolio_value': new_portfolio_value}
    
    continuous_env = ContinuousTradingEnv()
    
    def continuous_env_step_function(action_type, action=None):
        if action_type == 'reset':
            return continuous_env.reset()
        elif action_type == 'step':
            return continuous_env.step(action)
    
    sac_config = AgentConfig(
        agent_type=AgentType.SAC,
        state_dim=7,
        action_dim=1,  # 1 continuous action
        action_space=ActionSpace.CONTINUOUS,
        hidden_dim=128,
        learning_rate=0.0003,
        gamma=0.99,
        tau=0.005,
        epsilon=0.0,  # Not used in SAC
        epsilon_decay=1.0,
        epsilon_min=0.0,
        buffer_size=10000,
        batch_size=32,
        update_frequency=1,
        target_update_frequency=1
    )
    
    sac_agent = SACAgent(sac_config)
    training_env.register_agent("sac_trader", sac_agent)
    
    print("Training SAC agent...")
    for episode in range(30):  # Reduced for testing
        metrics = training_env.train_episode("sac_trader", continuous_env_step_function, max_steps=200)
        
        if episode % 5 == 0:
            print(f"Episode {episode}: Reward={metrics.total_reward:.4f}, "
                  f"Steps={metrics.episode_length}, Loss={metrics.loss:.6f}")
    
    sac_summary = training_env.get_training_summary("sac_trader")
    print(f"SAC Training Summary:")
    print(f"  Total episodes: {sac_summary['total_episodes']}")
    print(f"  Average recent reward: {sac_summary['average_reward_recent']:.4f}")
    print(f"  Best reward: {sac_summary['best_reward']:.4f}")
    
    # Compare agent performance
    print("\n=== Agent Performance Comparison ===")
    print(f"{'Agent':<12} {'Episodes':<10} {'Avg Reward':<12} {'Best Reward':<12}")
    print("-" * 50)
    
    for agent_name in ["dqn_trader", "ppo_trader", "sac_trader"]:
        summary = training_env.get_training_summary(agent_name)
        print(f"{agent_name:<12} "
              f"{summary['total_episodes']:<10} "
              f"{summary['average_reward_recent']:<12.4f} "
              f"{summary['best_reward']:<12.4f}")
    
    print("\nReinforcement learning agents testing completed successfully!")