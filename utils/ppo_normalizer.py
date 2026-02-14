"""
Running Mean/Std Normalizer for PPO.
Implements observation and reward normalization for stable training.
"""
import numpy as np
import torch
import os
from typing import Optional, Tuple


class RunningMeanStd:
    """
    Tracks running mean and standard deviation using Welford's algorithm.
    Used for observation normalization and reward scaling.
    """
    
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Small initial count to prevent division by zero
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray):
        """Update running statistics with a batch of data."""
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    
    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Update from batch statistics (Welford's online algorithm)."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)
    
    def normalize_torch(self, x: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        """Normalize a torch tensor using running statistics."""
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(np.sqrt(self.var + self.epsilon), dtype=torch.float32, device=device)
        return (x - mean) / std
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)

    def save(self, path: str):
        """Save statistics to a file."""
        np.savez(path, mean=self.mean, var=self.var, count=self.count)
    
    def load(self, path: str):
        """Load statistics from a file."""
        if not path.endswith('.npz'):
            path += '.npz'
        if os.path.exists(path):
            data = np.load(path)
            self.mean = data['mean']
            self.var = data['var']
            self.count = data['count']


class ObservationNormalizer:
    """
    Wraps RunningMeanStd for multi-agent observation normalization.
    Each agent's observation is normalized independently but shares statistics.
    """
    
    def __init__(self, obs_dim: int, epsilon: float = 1e-8):
        self.obs_dim = obs_dim
        self.rms = RunningMeanStd(shape=(obs_dim,), epsilon=epsilon)
        self.epsilon = epsilon
    
    def update(self, obs: np.ndarray):
        """
        Update with observations from all agents.
        
        Args:
            obs: [batch, n_agents, obs_dim] or [n_agents, obs_dim]
        """
        if obs.ndim == 3:
            # Flatten batch and agents
            obs_flat = obs.reshape(-1, self.obs_dim)
        elif obs.ndim == 2:
            obs_flat = obs
        else:
            raise ValueError(f"Expected 2D or 3D obs, got {obs.ndim}D")
        
        self.rms.update(obs_flat)
    
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations."""
        original_shape = obs.shape
        obs_flat = obs.reshape(-1, self.obs_dim)
        normalized = self.rms.normalize(obs_flat)
        return normalized.reshape(original_shape)
    
    def normalize_torch(self, obs: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        """Normalize a torch tensor of observations."""
        original_shape = obs.shape
        obs_flat = obs.reshape(-1, self.obs_dim)
        normalized = self.rms.normalize_torch(obs_flat, device)
        return normalized.reshape(original_shape)

    def save(self, base_path: str):
        self.rms.save(base_path + "_obs_rms.npz")
    
    def load(self, base_path: str):
        self.rms.load(base_path + "_obs_rms.npz")


class RewardNormalizer:
    """
    Normalizes rewards by dividing by running standard deviation of returns.
    Does NOT subtract mean (to preserve reward sign/magnitude signal).
    """
    
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(shape=())
        self.returns = None  # Will be initialized on first call
    
    def update(self, rewards: np.ndarray, dones: np.ndarray):
        """
        Update return statistics.
        
        Args:
            rewards: [n_agents] or [batch, n_agents]
            dones: [n_agents] or [batch, n_agents] boolean array
        """
        if rewards.ndim == 1:
            rewards = rewards.reshape(1, -1)
            dones = np.array([dones]).reshape(1, -1) if np.isscalar(dones) else dones.reshape(1, -1)
        
        batch_size, n_agents = rewards.shape
        
        if self.returns is None:
            self.returns = np.zeros(n_agents, dtype=np.float64)
        
        for i in range(batch_size):
            self.returns = self.returns * self.gamma * (1 - dones[i].astype(float)) + rewards[i]
            self.return_rms.update(self.returns.reshape(-1))
    
    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards by return std (no mean subtraction)."""
        return rewards / (self.return_rms.std + self.epsilon)
    
    @property
    def std(self) -> float:
        return self.return_rms.std

    def save(self, base_path: str):
        self.return_rms.save(base_path + "_reward_rms.npz")
    
    def load(self, base_path: str):
        self.return_rms.load(base_path + "_reward_rms.npz")
