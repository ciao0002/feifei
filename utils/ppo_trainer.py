"""
PPO Trainer for PPO-CoLight.
Implements PPO-Clip with value clipping, advantage normalization, and gradient clipping.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import numpy as np
from torch.distributions import Categorical

from .ppo_buffer import RolloutBuffer, compute_gae, MinibatchSampler


class PPOTrainer:
    """
    PPO Trainer with all stability tricks:
    - PPO-Clip for policy updates
    - Value Clipping (PPO2)
    - Advantage Normalization
    - Gradient Clipping
    - LR Annealing (NEW)
    - Target KL early stopping (NEW)
    """
    
    def __init__(self,
                 network: nn.Module,
                 lr: float = 3e-4,
                 gamma: float = 0.95,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.05, # Higher start for better exploration
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 minibatch_size: int = 64,
                 device: str = 'cpu',
                 lr_end: float = 1e-4,
                 entropy_coef_end: float = 0.01, # Linear decay to stable policy
                 total_rounds: int = 100,
                 target_kl: Optional[float] = None,
                 use_contrastive_aug: bool = False,
                 contrastive_coef: float = 0.0,
                 aug_noise_std: float = 0.01,
                 lane_feature_slice: Optional[Tuple[int, int]] = None,
                 use_pred_aux: bool = False,
                 pred_aux_coef: float = 0.0,
                 pred_feature_slice: Optional[Tuple[int, int]] = None):
        
        self.network = network
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef_start = entropy_coef
        self.entropy_coef_end = entropy_coef_end
        self.entropy_coef = entropy_coef # Current value
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        
        # LR Annealing
        self.lr_start = lr
        self.lr_end = lr_end
        self.total_rounds = total_rounds
        self.current_round = 0
        self.target_kl = target_kl
        self.use_contrastive_aug = use_contrastive_aug
        self.contrastive_coef = float(contrastive_coef)
        self.aug_noise_std = float(aug_noise_std)
        self.lane_feature_slice = lane_feature_slice
        self.use_pred_aux = use_pred_aux
        self.pred_aux_coef = float(pred_aux_coef)
        self.pred_feature_slice = pred_feature_slice
        
        # Optimizer
        self.optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)

    def _augment_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise augmentation on lane_num_vehicle slice (or all dims fallback).
        """
        obs_aug = obs.clone()
        if self.aug_noise_std <= 0:
            return obs_aug

        if self.lane_feature_slice is not None:
            start, end = self.lane_feature_slice
            noise = torch.randn_like(obs_aug[..., start:end]) * self.aug_noise_std
            obs_aug[..., start:end] = torch.clamp(obs_aug[..., start:end] + noise, min=0.0)
        else:
            noise = torch.randn_like(obs_aug) * self.aug_noise_std
            obs_aug = torch.clamp(obs_aug + noise, min=0.0)
        return obs_aug
    
    def update_hyperparams(self, current_round: int):
        """Update learning rate and entropy coefficient using linear annealing."""
        self.current_round = current_round
        progress = min(1.0, current_round / max(1, self.total_rounds))
        
        # LR annealing
        new_lr = self.lr_start + (self.lr_end - self.lr_start) * progress
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        # Entropy annealing
        self.entropy_coef = self.entropy_coef_start + (self.entropy_coef_end - self.entropy_coef_start) * progress
        
        return new_lr, self.entropy_coef

    
    def train(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform PPO update on collected rollout.
        
        Args:
            buffer: RolloutBuffer with collected trajectory
        
        Returns:
            dict of training metrics
        """
        # Compute GAE with proper time-limit bootstrap
        def critic_fn(obs, adj):
            with torch.no_grad():
                _, values = self.network(obs, adj)
            return values
        
        advantages, returns = compute_gae(
            buffer, 
            critic_fn,
            gamma=self.gamma,
            lam=self.gae_lambda
        )
        
        # Move to device
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_contrastive_loss = 0.0
        total_pred_loss = 0.0
        n_updates = 0
        
        # Multiple epochs of updates
        stop_update = False
        for epoch in range(self.n_epochs):
            sampler = MinibatchSampler(
                buffer, advantages, returns,
                minibatch_size=self.minibatch_size,
                shuffle=True
            )
            
            for batch in sampler:
                # Move batch to device
                obs = batch['obs'].to(self.device)
                adj = batch['adj'].to(self.device)
                actions = batch['actions'].to(self.device).long()
                log_probs_old = batch['log_probs_old'].to(self.device)
                values_old = batch['values_old'].to(self.device)
                batch_advantages = batch['advantages'].to(self.device)
                batch_returns = batch['returns'].to(self.device)
                batch_next_obs = batch['next_obs'].to(self.device)
                
                # Get current policy outputs with explicit backbone features
                # batch['obs'] is [batch_T, N, obs_dim]
                h = self.network.forward_backbone(obs, adj)
                logits = self.network.get_actor_output(h)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy()
                values = self.network.get_critic_output(h, adj)
                
                # PPO-Clip objective
                ratio = torch.exp(log_probs - log_probs_old)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping (PPO2)
                values_clipped = values_old + torch.clamp(
                    values - values_old, -self.clip_range, self.clip_range
                )
                value_loss1 = (values - batch_returns) ** 2
                value_loss2 = (values_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Contrastive consistency loss on augmented observations
                if self.use_contrastive_aug and self.contrastive_coef > 0:
                    obs_aug = self._augment_obs(obs)
                    h_aug = self.network.forward_backbone(obs_aug, adj)
                    contrastive_loss = nn.functional.mse_loss(h, h_aug)
                else:
                    contrastive_loss = torch.zeros((), device=self.device)

                # Predictive auxiliary loss: predict next-step lane feature from (h_t, a_t)
                if self.use_pred_aux and self.pred_aux_coef > 0 and getattr(self.network, "use_pred_aux", False):
                    pred_next = self.network.get_pred_output(h, actions)
                    if self.pred_feature_slice is not None:
                        start, end = self.pred_feature_slice
                        target_next = batch_next_obs[..., start:end]
                    else:
                        target_next = batch_next_obs[..., :pred_next.shape[-1]]
                    pred_loss = nn.functional.mse_loss(pred_next, target_next)
                else:
                    pred_loss = torch.zeros((), device=self.device)
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.value_coef * value_loss 
                    + self.entropy_coef * entropy_loss
                    + self.contrastive_coef * contrastive_loss
                    + self.pred_aux_coef * pred_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    approx_kl = (log_probs_old - log_probs).mean().item()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                total_contrastive_loss += contrastive_loss.item()
                total_pred_loss += pred_loss.item()
                n_updates += 1
                
                # Target KL early stopping
                if self.target_kl is not None and abs(approx_kl) > self.target_kl:
                    print(f"  [Early Stop] KL={abs(approx_kl):.4f} > target={self.target_kl:.4f}, stopping epoch {epoch}")
                    stop_update = True
                    break

                
                # === Debug diagnostics (first batch of each epoch) ===
                if n_updates == 1 or (n_updates - 1) % 10 == 0:
                    with torch.no_grad():
                        ratio_flat = ratio.flatten()
                        adv_flat = batch_advantages.flatten()
                        print(f"  [DEBUG] ratio: min={ratio_flat.min():.4f}, max={ratio_flat.max():.4f}, "
                              f"p50={ratio_flat.median():.4f}, p99={ratio_flat.quantile(0.99):.4f}")
                        print(f"  [DEBUG] advantage: min={adv_flat.min():.4f}, max={adv_flat.max():.4f}, "
                              f"mean={adv_flat.mean():.4f}, std={adv_flat.std():.4f}")
                        print(f"  [DEBUG] log_prob diff: mean={(log_probs - log_probs_old).mean():.6f}")
            if stop_update:
                break
        
        # Return average metrics
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'approx_kl': total_approx_kl / n_updates,
            'clip_fraction': total_clip_fraction / n_updates,
            'contrastive_loss': total_contrastive_loss / n_updates,
            'pred_loss': total_pred_loss / n_updates,
            'n_updates': n_updates,
        }


class PPORolloutCollector:
    """
    Collects on-policy rollouts for PPO training.
    """
    
    def __init__(self, 
                 agent,
                 env,
                 buffer: RolloutBuffer,
                 device: str = 'cpu'):
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.device = device
    
    def collect_rollout(self, n_steps: int) -> Tuple[float, int]:
        """
        Collect n_steps of experience.
        
        Returns:
            total_reward: sum of rewards
            n_steps_collected: actual steps collected
        """
        self.buffer.reset()
        
        state = self.env.reset()
        total_reward = 0.0
        
        for step in range(n_steps):
            # Convert state to tensor
            obs, adj = self.agent.convert_state_to_tensor(state)
            
            # Get action from policy
            with torch.no_grad():
                actions, log_probs, entropy, values = self.agent.network.get_action(obs, adj)
            
            # Execute in environment
            action_list = actions.cpu().numpy()[0]  # [agent]
            next_state, reward, done, _ = self.env.step(action_list)
            
            # Convert reward to tensor
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            
            # Determine if time-limit (last step)
            time_limit = (step == n_steps - 1)
            
            # Store in buffer (tensor snapshots!)
            self.buffer.add(
                obs_tensor=obs.squeeze(0),  # [agent, obs_dim]
                adj_tensor=adj.squeeze(0),  # [agent, neighbor, agent]
                action=actions.squeeze(0),
                log_prob=log_probs.squeeze(0),
                value=values.squeeze(0),
                reward=reward_tensor,
                done=done,
                time_limit=time_limit
            )
            
            total_reward += sum(reward)
            state = next_state
            
            if done:
                break
        
        # CRITICAL: Store last state for bootstrap
        obs_last, adj_last = self.agent.convert_state_to_tensor(state)
        self.buffer.set_last_state(obs_last.squeeze(0), adj_last.squeeze(0))
        
        return total_reward, len(self.buffer)
