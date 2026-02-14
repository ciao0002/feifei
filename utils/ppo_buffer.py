"""
PPO Rollout Buffer for on-policy training.
Stores network input tensors directly (not state dicts) to ensure log_prob consistency.
"""
import torch
import numpy as np
from typing import List, Optional, Tuple


class RolloutBuffer:
    """
    On-policy rollout buffer that stores network input tensor snapshots.
    
    Key design:
    - Stores obs_tensors (network inputs), NOT raw state dicts
    - Update时不重算输入, 直接用rollout时存的张量
    - Supports time-limit bootstrap
    """
    
    def __init__(self, n_agents: int, device: str = 'cpu'):
        self.n_agents = n_agents
        self.device = device
        self.reset()
    
    def reset(self):
        """Clear buffer for new rollout"""
        self.obs_tensors: List[torch.Tensor] = []      # [T, N_agents, obs_dim]
        self.adj_tensors: List[torch.Tensor] = []      # [T, N_agents, K, N_agents]
        self.actions: List[torch.Tensor] = []          # [T, N_agents]
        self.log_probs_old: List[torch.Tensor] = []    # [T, N_agents]
        self.values_old: List[torch.Tensor] = []       # [T, N_agents]
        self.rewards: List[torch.Tensor] = []          # [T, N_agents]
        self.dones: List[bool] = []                    # [T]
        self.time_limits: List[bool] = []              # [T]
        
        # For time-limit bootstrap
        self.last_obs_tensor: Optional[torch.Tensor] = None
        self.last_adj_tensor: Optional[torch.Tensor] = None
    
    def add(self, 
            obs_tensor: torch.Tensor, 
            adj_tensor: torch.Tensor,
            action: torch.Tensor,
            log_prob: torch.Tensor,
            value: torch.Tensor,
            reward: torch.Tensor,
            done: bool,
            time_limit: bool = False):
        """
        Add one step to buffer.
        
        CRITICAL: 存的是网络已经处理好的张量，detach+clone断开计算图
        """
        self.obs_tensors.append(obs_tensor.detach().clone())
        self.adj_tensors.append(adj_tensor.detach().clone())
        self.actions.append(action.detach().clone())
        self.log_probs_old.append(log_prob.detach().clone())
        self.values_old.append(value.detach().clone())
        self.rewards.append(reward.detach().clone() if isinstance(reward, torch.Tensor) 
                           else torch.tensor(reward, dtype=torch.float32))
        self.dones.append(done)
        self.time_limits.append(time_limit)
    
    def set_last_state(self, obs_tensor: torch.Tensor, adj_tensor: torch.Tensor):
        """
        Store the final state for time-limit bootstrap.
        MUST be called at end of rollout.
        """
        self.last_obs_tensor = obs_tensor.detach().clone()
        self.last_adj_tensor = adj_tensor.detach().clone()
    
    def __len__(self):
        return len(self.rewards)
    
    def get_tensors(self) -> Tuple[torch.Tensor, ...]:
        """Stack all tensors for batch processing"""
        return (
            torch.stack(self.obs_tensors),      # [T, N_agents, obs_dim]
            torch.stack(self.adj_tensors),      # [T, N_agents, K, N_agents]
            torch.stack(self.actions),          # [T, N_agents]
            torch.stack(self.log_probs_old),    # [T, N_agents]
            torch.stack(self.values_old),       # [T, N_agents]
            torch.stack(self.rewards),          # [T, N_agents]
        )


def compute_gae(buffer: RolloutBuffer, 
                critic_fn,
                gamma: float = 0.95, 
                lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE with proper time-limit bootstrap.
    
    交通场景: 3600s后强制结束，但这不是true terminal
    必须用 V(s_{T+1}) bootstrap，不能置0
    
    Args:
        buffer: RolloutBuffer with collected trajectory
        critic_fn: function that takes (obs, adj) and returns V values
        gamma: discount factor
        lam: GAE lambda
    
    Returns:
        advantages: [T, N_agents] normalized advantages
        returns: [T, N_agents] target returns for value function
    """
    T = len(buffer)
    n_agents = buffer.n_agents
    
    advantages = torch.zeros(T, n_agents)
    returns = torch.zeros(T, n_agents)
    
    # Get values from buffer
    values_old = torch.stack(buffer.values_old)  # [T, N_agents]
    rewards = torch.stack(buffer.rewards)        # [T, N_agents]
    
    # CRITICAL: Get last_value for bootstrap
    if buffer.time_limits[-1]:
        # Time-limit结束: 用critic评估s_{T+1}
        assert buffer.last_obs_tensor is not None, \
            "Must call buffer.set_last_state() at end of rollout!"
        with torch.no_grad():
            # Add batch dimension: [agent, dim] -> [1, agent, dim]
            last_obs = buffer.last_obs_tensor.unsqueeze(0)
            last_adj = buffer.last_adj_tensor.unsqueeze(0)
            last_value = critic_fn(last_obs, last_adj)
            last_value = last_value.squeeze(0)  # [1, agent] -> [agent]
    else:
        # True terminal (交通场景基本不会有)
        last_value = torch.zeros(n_agents)
    
    # Standard GAE计算
    next_advantage = torch.zeros(n_agents)
    
    for t in reversed(range(T)):
        current_value = values_old[t]  # [N_agents]
        
        # TD error: r + γV(s') - V(s)
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values_old[t + 1]
        
        delta = rewards[t] + gamma * next_value - current_value
        
        # GAE: A_t = δ_t + γλA_{t+1}
        advantages[t] = delta + gamma * lam * next_advantage
        returns[t] = advantages[t] + current_value
        
        next_advantage = advantages[t]
    
    # Advantage Normalization
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


class MinibatchSampler:
    """Generate minibatches from rollout buffer for PPO updates"""
    
    def __init__(self, 
                 buffer: RolloutBuffer,
                 advantages: torch.Tensor,
                 returns: torch.Tensor,
                 minibatch_size: int = 32, # Batch by time steps
                 shuffle: bool = True):
        self.buffer = buffer
        self.advantages = advantages
        self.returns = returns
        self.minibatch_size = minibatch_size
        self.shuffle = shuffle
        
        # Batch along Time dimension ONLY to keep agent relationships for GAT
        self.T = len(buffer)
        self.n_agents = buffer.n_agents
        self.total_samples = self.T
    
    def __iter__(self):
        # Get all data as tensors: [T, N_agents, ...]
        obs, adj, actions, log_probs_old, values_old, rewards = self.buffer.get_tensors()
        
        # Generate time-step indices
        indices = np.arange(self.T)
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Yield minibatches of time-steps
        for start in range(0, self.T, self.minibatch_size):
            end = min(start + self.minibatch_size, self.T)
            batch_indices = torch.LongTensor(indices[start:end])
            next_indices = torch.clamp(batch_indices + 1, max=self.T - 1)
            next_obs = obs[next_indices].clone()
            next_adj = adj[next_indices].clone()

            # Last rollout step should bootstrap from stored final state if available.
            if self.buffer.last_obs_tensor is not None and self.buffer.last_adj_tensor is not None:
                mask_last = (batch_indices == (self.T - 1))
                if mask_last.any():
                    next_obs[mask_last] = self.buffer.last_obs_tensor
                    next_adj[mask_last] = self.buffer.last_adj_tensor
            
            yield {
                'obs': obs[batch_indices],           # [batch_T, N, obs_dim]
                'adj': adj[batch_indices],           # [batch_T, N, K, N]
                'next_obs': next_obs,                # [batch_T, N, obs_dim]
                'next_adj': next_adj,                # [batch_T, N, K, N]
                'actions': actions[batch_indices],   # [batch_T, N]
                'log_probs_old': log_probs_old[batch_indices],
                'values_old': values_old[batch_indices],
                'advantages': self.advantages[batch_indices], # [batch_T, N]
                'returns': self.returns[batch_indices],
            }
