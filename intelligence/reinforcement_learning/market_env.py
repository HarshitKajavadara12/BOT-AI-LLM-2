"""
Market Environment for Reinforcement Learning
Trading environment following OpenAI Gym interface
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd


class MarketEnvironment(gym.Env):
    """Trading environment for RL agents."""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001, lookback_window: int = 50):
        super(MarketEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # State space: [balance, position, market_features]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lookback_window + 2,), dtype=np.float32
        )
        
        # Action space: [-1, 1] (sell, hold, buy)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.total_reward = 0.0
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_reward = 0.0
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute trade
        action_value = action[0]
        trade_cost = abs(action_value) * self.transaction_cost
        
        # Calculate PnL
        prev_value = self.balance + self.position * current_price
        
        # Update position
        trade_amount = action_value * self.balance / current_price
        self.position += trade_amount
        self.balance -= trade_amount * current_price * (1 + self.transaction_cost)
        
        # Calculate new value
        new_value = self.balance + self.position * current_price
        reward = (new_value - prev_value) / prev_value
        
        self.total_reward += reward
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1) or (self.balance <= 0)
        
        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': new_value
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Market features
        market_data = self.data.iloc[
            self.current_step - self.lookback_window:self.current_step
        ]['close'].values
        
        # Normalize
        market_data = (market_data - market_data.mean()) / (market_data.std() + 1e-8)
        
        # Add balance and position
        obs = np.concatenate([
            [self.balance / self.initial_balance],
            [self.position],
            market_data
        ])
        
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        """Render the environment."""
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
              f"Position: {self.position:.2f}, Total Reward: {self.total_reward:.4f}")


class MultiAssetEnvironment(gym.Env):
    """Multi-asset trading environment."""
    
    def __init__(self, data: Dict[str, pd.DataFrame], initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001, lookback_window: int = 50):
        super(MultiAssetEnvironment, self).__init__()
        
        self.data = data
        self.assets = list(data.keys())
        self.num_assets = len(self.assets)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # State space: [balance, positions, market_features]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1 + self.num_assets + self.num_assets * lookback_window,),
            dtype=np.float32
        )
        
        # Action space: portfolio weights for each asset
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_assets,), dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = initial_balance
        self.positions = np.zeros(self.num_assets)
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.positions = np.zeros(self.num_assets)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step."""
        self.current_step += 1
        
        # Normalize action to weights that sum to 1
        weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        
        # Get current prices
        current_prices = np.array([
            self.data[asset].iloc[self.current_step]['close']
            for asset in self.assets
        ])
        
        # Calculate current portfolio value
        prev_value = self.balance + np.sum(self.positions * current_prices)
        
        # Rebalance portfolio
        target_positions = weights * prev_value / current_prices
        trades = target_positions - self.positions
        
        # Apply transaction costs
        trade_costs = np.abs(trades * current_prices) * self.transaction_cost
        total_cost = np.sum(trade_costs)
        
        self.positions = target_positions
        self.balance -= total_cost
        
        # Calculate new portfolio value
        new_value = self.balance + np.sum(self.positions * current_prices)
        reward = (new_value - prev_value) / prev_value
        
        # Check if done
        done = (self.current_step >= min(len(df) for df in self.data.values()) - 1)
        
        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'positions': self.positions.copy(),
            'portfolio_value': new_value
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        market_features = []
        
        for asset in self.assets:
            data = self.data[asset].iloc[
                self.current_step - self.lookback_window:self.current_step
            ]['close'].values
            
            # Normalize
            data = (data - data.mean()) / (data.std() + 1e-8)
            market_features.extend(data)
        
        obs = np.concatenate([
            [self.balance / self.initial_balance],
            self.positions,
            market_features
        ])
        
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        """Render environment."""
        current_prices = np.array([
            self.data[asset].iloc[self.current_step]['close']
            for asset in self.assets
        ])
        portfolio_value = self.balance + np.sum(self.positions * current_prices)
        
        print(f"Step: {self.current_step}, Portfolio Value: {portfolio_value:.2f}")
        for i, asset in enumerate(self.assets):
            print(f"  {asset}: Position={self.positions[i]:.2f}, "
                  f"Price={current_prices[i]:.2f}")
