"""
PPO-CoLight Agent with Actor-Critic architecture.
Based on CoLight's GAT structure with PPO training.
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict, List

# Import base agent structure
from .agent import Agent


class RepeatVector3D(nn.Module):
    """Repeat tensor along a new dimension (PyTorch version of Keras layer)"""
    def __init__(self, times: int):
        super().__init__()
        self.times = times
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, agent, dim] -> [batch, 1, agent, dim] -> [batch, agent, agent, dim]
        return x.unsqueeze(1).repeat(1, self.times, 1, 1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention for GAT (CoLight style)"""
    
    def __init__(self, 
                 num_agents: int,
                 num_neighbors: int,
                 d_in: int = 32,
                 h_dim: int = 16,
                 d_out: int = 32,
                 num_heads: int = 5,
                 use_edge_feature: bool = False,
                 num_row: int = 0,
                 num_col: int = 0,
                 edge_hidden_dim: int = 16):
        super().__init__()
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.num_heads = num_heads
        self.h_dim = h_dim
        self.use_edge_feature = use_edge_feature
        self.num_row = max(int(num_row), 1)
        self.num_col = max(int(num_col), 1)
        
        # Attention projections
        self.agent_proj = nn.Linear(d_in, h_dim * num_heads)
        self.neighbor_proj = nn.Linear(d_in, h_dim * num_heads)
        self.neighbor_value_proj = nn.Linear(d_in, h_dim * num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(h_dim, d_out)

        if self.use_edge_feature:
            # Edge feature: [same_axis, diff_axis, normalized_distance, connected_mask]
            self.edge_mlp = nn.Sequential(
                nn.Linear(4, edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim, num_heads),
            )
            coords = []
            for idx in range(self.num_agents):
                r = idx // self.num_col
                c = idx % self.num_col
                coords.append([float(r), float(c)])
            self.register_buffer("agent_coords", torch.tensor(coords, dtype=torch.float32), persistent=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, 
                in_feats: torch.Tensor, 
                adj_matrix: torch.Tensor,
                phase_axis: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            in_feats: [batch, agent, dim]
            adj_matrix: [batch, agent, neighbor, agent] one-hot adjacency
        
        Returns:
            out: [batch, agent, d_out]
        """
        batch_size = in_feats.shape[0]
        
        # Agent representation: [batch, agent, 1, dim]
        agent_repr = in_feats.unsqueeze(2)
        
        # Neighbor representation: [batch, agent, agent, dim]
        neighbor_repr = in_feats.unsqueeze(1).repeat(1, self.num_agents, 1, 1)
        
        # Apply adjacency to get neighbors: [batch, agent, neighbor, dim]
        neighbor_repr = torch.matmul(adj_matrix, neighbor_repr)
        
        # Project to heads
        # [batch, agent, 1, h_dim * num_heads] -> [batch, agent, num_heads, 1, h_dim]
        agent_head = self.agent_proj(agent_repr)
        agent_head = agent_head.view(batch_size, self.num_agents, 1, self.h_dim, self.num_heads)
        agent_head = agent_head.permute(0, 1, 4, 2, 3)
        
        # [batch, agent, neighbor, h_dim * num_heads] -> [batch, agent, num_heads, neighbor, h_dim]
        neighbor_head = self.neighbor_proj(neighbor_repr)
        neighbor_head = neighbor_head.view(batch_size, self.num_agents, self.num_neighbors, self.h_dim, self.num_heads)
        neighbor_head = neighbor_head.permute(0, 1, 4, 2, 3)
        
        # Attention logits: [batch, agent, heads, 1, neighbor]
        att_logits = torch.matmul(agent_head, neighbor_head.transpose(-1, -2))

        # CityLight-style edge-aware attention bias (relation + distance).
        if self.use_edge_feature and phase_axis is not None:
            # [batch, agent, neighbor] -> neighbor index in global agent space
            neighbor_idx = torch.argmax(adj_matrix, dim=-1).long()

            # Build ego and neighbor coordinates
            batch_size_local = in_feats.shape[0]
            ego_idx = torch.arange(self.num_agents, device=in_feats.device).view(1, self.num_agents, 1)
            ego_idx = ego_idx.expand(batch_size_local, -1, self.num_neighbors).long()

            coords = self.agent_coords.to(in_feats.device)
            ego_rc = coords[ego_idx]          # [batch, agent, neighbor, 2]
            neighbor_rc = coords[neighbor_idx]  # [batch, agent, neighbor, 2]

            dr = neighbor_rc[..., 0] - ego_rc[..., 0]
            dc = neighbor_rc[..., 1] - ego_rc[..., 1]
            manhattan = dr.abs() + dc.abs()
            denom = max(self.num_row + self.num_col - 2, 1)
            dist_norm = manhattan / float(denom)

            # Orientation proxy: horizontal if |dc| >= |dr|, else vertical.
            neighbor_horizontal = (dc.abs() >= dr.abs()).float()
            phase_horizontal = phase_axis.unsqueeze(-1).expand_as(neighbor_horizontal)
            same_axis = (neighbor_horizontal == phase_horizontal).float()
            connected = (manhattan > 0).float()

            edge_raw = torch.stack(
                [same_axis, 1.0 - same_axis, dist_norm, connected],
                dim=-1,
            )  # [batch, agent, neighbor, 4]

            edge_bias = self.edge_mlp(edge_raw)  # [batch, agent, neighbor, heads]
            edge_bias = edge_bias.permute(0, 1, 3, 2).unsqueeze(3)  # [batch, agent, heads, 1, neighbor]
            att_logits = att_logits + edge_bias

        att = F.softmax(att_logits, dim=-1)
        
        # Value projection
        neighbor_value = self.neighbor_value_proj(neighbor_repr)
        neighbor_value = neighbor_value.view(batch_size, self.num_agents, self.num_neighbors, self.h_dim, self.num_heads)
        neighbor_value = neighbor_value.permute(0, 1, 4, 2, 3)
        
        # Weighted sum: [batch, agent, heads, 1, h_dim]
        out = torch.matmul(att, neighbor_value)
        
        # Average over heads and reshape: [batch, agent, h_dim]
        out = out.mean(dim=2).squeeze(2)
        
        # Output projection
        out = F.relu(self.out_proj(out))
        
        return out


class InputEmbeddingBlock(nn.Module):
    """Input embedding block: MLP + LayerNorm before GAT backbone."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.norm(h)
        return h


class PhaseCompetitionHead(nn.Module):
    """
    Phase competition actor head.

    Builds per-phase embeddings from (intersection embedding + phase id),
    runs self-attention across phases to model competition, then outputs logits.
    """

    def __init__(
        self,
        backbone_dim: int,
        num_actions: int,
        phase_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions

        self.phase_embed = nn.Sequential(
            nn.Linear(backbone_dim + num_actions, phase_dim),
            nn.ReLU(),
            nn.LayerNorm(phase_dim),
        )

        # Attention across phases (per intersection).
        self.attn = nn.MultiheadAttention(
            embed_dim=phase_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.out = nn.Sequential(
            nn.Linear(phase_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Cache identity matrix for one-hot phase id.
        self.register_buffer("phase_eye", torch.eye(num_actions), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, agent, backbone_dim]
        Returns:
            logits: [batch, agent, num_actions]
        """
        batch_size, num_agents, _ = h.shape

        # [num_actions, num_actions] -> [1, 1, num_actions, num_actions]
        phase_onehot = self.phase_eye.view(1, 1, self.num_actions, self.num_actions)
        phase_onehot = phase_onehot.expand(batch_size, num_agents, -1, -1)

        # Broadcast h to phases: [batch, agent, num_actions, backbone_dim]
        h_rep = h.unsqueeze(2).expand(-1, -1, self.num_actions, -1)
        x = torch.cat([h_rep, phase_onehot], dim=-1)

        # Phase embeddings: [batch, agent, num_actions, phase_dim]
        phase_emb = self.phase_embed(x)

        # Self-attention over phases; flatten (batch*agent) for efficiency.
        phase_flat = phase_emb.reshape(batch_size * num_agents, self.num_actions, -1)
        attn_out, _ = self.attn(phase_flat, phase_flat, phase_flat, need_weights=False)
        attn_out = attn_out.reshape(batch_size, num_agents, self.num_actions, -1)

        logits = self.out(attn_out).squeeze(-1)
        return logits


class PPOCoLightNetwork(nn.Module):
    """
    Actor-Critic network with Spatial-GAT (STABLE VERSION).
    
    Architecture:
    1. MLP Embedding
    2. Spatial GAT (1 layer) -> Global context
    3. Actor head (MLP) -> Softmax
    4. Critic head (MLP) -> Value
    """
    
    def __init__(self,
                 num_agents: int,
                 num_neighbors: int,
                 obs_dim: int,
                 num_actions: int,
                 mlp_layers: List[int] = [32, 32],
                 gat_layers: List[Tuple[int, int]] = [(16, 32)],
                 num_heads: int = 5,
                 use_input_embed_block: bool = False,
                 input_embed_hidden: int = 64,
                 input_embed_dim: int = 128,
                 use_phase_comp_head: bool = False,
                 phase_comp_dim: int = 64,
                 phase_comp_heads: int = 4,
                 use_pred_aux: bool = False,
                 pred_aux_hidden: int = 64,
                 pred_aux_out_dim: int = 12,
                 use_edge_feature_gat: bool = False,
                 num_row: int = 0,
                 num_col: int = 0,
                 edge_feature_hidden: int = 16):
        super().__init__()
        
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.num_actions = num_actions
        self.use_input_embed_block = use_input_embed_block
        self.use_phase_comp_head = use_phase_comp_head
        self.use_pred_aux = use_pred_aux
        self.pred_aux_out_dim = pred_aux_out_dim
        self.use_edge_feature_gat = use_edge_feature_gat

        if self.use_input_embed_block:
            self.input_embed_block = InputEmbeddingBlock(
                input_dim=obs_dim,
                hidden_dim=input_embed_hidden,
                output_dim=input_embed_dim,
            )
            prev_dim = input_embed_dim
        else:
            prev_dim = obs_dim
        
        # ========== Shared Backbone ==========
        self.embed_layers = nn.ModuleList()
        self.embed_norms = nn.ModuleList()
        for layer_dim in mlp_layers:
            self.embed_layers.append(nn.Linear(prev_dim, layer_dim))
            self.embed_norms.append(nn.LayerNorm(layer_dim))
            prev_dim = layer_dim
        
        # GAT layers (Spatial coordination)
        self.gat_layers = nn.ModuleList()
        gat_in_dim = mlp_layers[-1]
        for h_dim, out_dim in gat_layers:
            self.gat_layers.append(
                MultiHeadAttention(
                    num_agents=num_agents,
                    num_neighbors=num_neighbors,
                    d_in=gat_in_dim,
                    h_dim=h_dim,
                    d_out=out_dim,
                    num_heads=num_heads,
                    use_edge_feature=use_edge_feature_gat,
                    num_row=num_row,
                    num_col=num_col,
                    edge_hidden_dim=edge_feature_hidden,
                )
            )
            gat_in_dim = out_dim
        
        self.backbone_out_dim = gat_in_dim
        
        # ========== Actor Head (Standard MLP) ==========
        if self.use_phase_comp_head:
            self.actor_head = PhaseCompetitionHead(
                backbone_dim=self.backbone_out_dim,
                num_actions=self.num_actions,
                phase_dim=phase_comp_dim,
                num_heads=phase_comp_heads,
            )
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(self.backbone_out_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_actions)
            )
        
        # ========== Critic Head (Standard MLP) ==========
        # Using self features + neighbor pooled features
        self.critic_head = nn.Sequential(
            nn.Linear(self.backbone_out_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # ========== Predictive Auxiliary Head ==========
        # Predict next-step lane feature from (backbone feature + action onehot)
        if self.use_pred_aux:
            self.pred_aux_head = nn.Sequential(
                nn.Linear(self.backbone_out_dim + self.num_actions, pred_aux_hidden),
                nn.ReLU(),
                nn.Linear(pred_aux_hidden, self.pred_aux_out_dim)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Special init for actor output (smaller for exploration)
        if not self.use_phase_comp_head:
            nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        
        # Special init for critic output
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
    
    def forward_backbone(self, 
                         obs: torch.Tensor, 
                         adj: torch.Tensor) -> torch.Tensor:
        """Shared backbone: MLP + GAT"""
        h = obs
        phase_axis = None
        if self.use_edge_feature_gat:
            phase_axis = self._infer_phase_axis(obs)
        if self.use_input_embed_block:
            h = self.input_embed_block(h)

        # MLP with LayerNorm
        for linear, norm in zip(self.embed_layers, self.embed_norms):
            h = F.relu(norm(linear(h)))
        
        # GAT (Spatial coordination)
        for gat in self.gat_layers:
            h = gat(h, adj, phase_axis=phase_axis)
        
        return h

    @staticmethod
    def _infer_phase_axis(obs: torch.Tensor) -> torch.Tensor:
        """
        Infer current phase axis from the first 8 phase bits.
        Returns:
            phase_axis: [batch, agent], horizontal=1.0, vertical=0.0
        """
        if obs.shape[-1] < 8:
            return torch.ones(obs.shape[0], obs.shape[1], device=obs.device)

        # Templates follow current project PHASE encoding.
        # phase_1/3 -> horizontal, phase_2/4 -> vertical
        templates = torch.tensor(
            [
                [0, 1, 0, 1, 0, 0, 0, 0],  # phase 1
                [0, 0, 0, 0, 0, 1, 0, 1],  # phase 2
                [1, 0, 1, 0, 0, 0, 0, 0],  # phase 3
                [0, 0, 0, 0, 1, 0, 1, 0],  # phase 4
            ],
            dtype=obs.dtype,
            device=obs.device,
        )
        phase_bits = obs[..., :8]
        scores = torch.einsum("bad,pd->bap", phase_bits, templates)
        phase_idx = torch.argmax(scores, dim=-1) + 1
        phase_axis = ((phase_idx == 1) | (phase_idx == 3)).float()
        return phase_axis
    
    def get_actor_output(self, h: torch.Tensor) -> torch.Tensor:
        """Standard Actor MLP"""
        logits = self.actor_head(h).squeeze(-1)
        return logits
    
    def get_critic_output(self, 
                          h: torch.Tensor, 
                          adj: torch.Tensor) -> torch.Tensor:
        """Critic with adjacency-masked neighbor pooling"""
        # Neighbor pooling using adjacency mask
        h_expanded = h.unsqueeze(1).repeat(1, self.num_agents, 1, 1)
        h_neighbors = torch.matmul(adj, h_expanded)
        
        neighbor_count = adj.sum(dim=-1).clamp(min=1)
        h_pool = h_neighbors.sum(dim=2) / neighbor_count.sum(dim=-1, keepdim=True).clamp(min=1)
        
        h_critic = torch.cat([h, h_pool], dim=-1)
        value = self.critic_head(h_critic).squeeze(-1)
        
        return value

    def get_pred_output(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predict next-step lane-related feature.

        Args:
            h: [batch, agent, backbone_dim]
            actions: [batch, agent]
        Returns:
            pred: [batch, agent, pred_aux_out_dim]
        """
        if not self.use_pred_aux:
            raise RuntimeError("Predictive auxiliary head is not enabled.")
        action_onehot = F.one_hot(actions.long(), num_classes=self.num_actions).float()
        pred_in = torch.cat([h, action_onehot], dim=-1)
        pred = self.pred_aux_head(pred_in)
        return pred
    
    def forward(self, 
                obs: torch.Tensor, 
                adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass"""
        h = self.forward_backbone(obs, adj)
        logits = self.get_actor_output(h)
        values = self.get_critic_output(h, adj)
        return logits, values
    
    def get_action(self,
                   obs: torch.Tensor,
                   adj: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, adj)
        dist = Categorical(logits=logits)
        
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return actions, log_probs, entropy, values
    
    def evaluate_actions(self,
                         obs: torch.Tensor,
                         adj: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, adj)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


class PPOCoLightAgent(Agent):
    """
    PPO-based CoLight Agent wrapper.
    Adapts PPOCoLightNetwork to the existing Agent interface.
    """
    
    def __init__(self, 
                 dic_agent_conf: Dict,
                 dic_traffic_env_conf: Dict,
                 dic_path: Dict,
                 cnt_round: int = 0,
                 intersection_id: str = "0"):
        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)
        self.num_actions = len(dic_traffic_env_conf["PHASE"])
        self.obs_lane_num_scale = float(dic_agent_conf.get("OBS_LANE_NUM_SCALE", 1.0))
        if self.obs_lane_num_scale <= 0:
            self.obs_lane_num_scale = 1.0
        self.obs_pressure_scale = float(dic_agent_conf.get("OBS_PRESSURE_SCALE", self.obs_lane_num_scale))
        if self.obs_pressure_scale <= 0:
            self.obs_pressure_scale = self.obs_lane_num_scale
        self.use_phase_reindex_obs = bool(dic_agent_conf.get("USE_PHASE_REINDEX_OBS", False))
        self.obs_dim = self._cal_obs_dim()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build network
        self.network = PPOCoLightNetwork(
            num_agents=self.num_agents,
            num_neighbors=self.num_neighbors,
            obs_dim=self.obs_dim,
            num_actions=self.num_actions,
            mlp_layers=dic_agent_conf.get('MLP_LAYERS', [32, 32]),
            gat_layers=dic_agent_conf.get('CNN_layers', [(16, 32)]),
            num_heads=dic_agent_conf.get('NUM_HEADS', 5),
            use_input_embed_block=dic_agent_conf.get('USE_INPUT_EMBED_BLOCK', False),
            input_embed_hidden=dic_agent_conf.get('INPUT_EMBED_HIDDEN', 64),
            input_embed_dim=dic_agent_conf.get('INPUT_EMBED_DIM', 128),
            use_phase_comp_head=dic_agent_conf.get('USE_PHASE_COMP_HEAD', False),
            phase_comp_dim=dic_agent_conf.get('PHASE_COMP_DIM', 64),
            phase_comp_heads=dic_agent_conf.get('PHASE_COMP_HEADS', 4),
            use_pred_aux=dic_agent_conf.get('USE_PRED_AUX', False),
            pred_aux_hidden=dic_agent_conf.get('PRED_AUX_HIDDEN', 64),
            pred_aux_out_dim=dic_agent_conf.get('PRED_AUX_OUT_DIM', 12),
            use_edge_feature_gat=dic_agent_conf.get('USE_EDGE_FEATURE_GAT', False),
            num_row=dic_traffic_env_conf.get('NUM_ROW', 0),
            num_col=dic_traffic_env_conf.get('NUM_COL', 0),
            edge_feature_hidden=dic_agent_conf.get('EDGE_FEATURE_HIDDEN', 16),
        ).to(self.device)
        
        # Load weights if resuming
        if cnt_round > 0:
            self.load_network(f"round_{cnt_round - 1}")
    
    def _cal_obs_dim(self) -> int:
        """Calculate observation dimension based on features"""
        N = 0
        used_features = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        for feat_name in used_features:
            if feat_name == "adjacency_matrix":
                continue
            if "cur_phase" in feat_name:
                N += 8
            elif "traffic_movement" in feat_name:
                N += 12
            elif "lane_enter" in feat_name:
                N += 12
            elif "lane_num_vehicle" in feat_name:
                N += 12
            else:
                N += 12
        return N
    
    def convert_state_to_tensor(self, states: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert state dicts to network input tensors"""
        used_features = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        lane_scale_features = {
            "lane_num_vehicle",
            "lane_enter_num_vehicle_front",
        }
        pressure_feature = "traffic_movement_pressure_queue_efficient"
        
        obs_list = []
        adj_list = []
        
        for i in range(self.num_agents):
            obs_i = []
            phase_idx = 1
            if "cur_phase" in states[i] and len(states[i]["cur_phase"]) > 0:
                try:
                    phase_idx = int(states[i]["cur_phase"][0])
                except Exception:
                    phase_idx = 1

            for feature in used_features:
                if feature == "cur_phase":
                    if self.dic_traffic_env_conf.get("BINARY_PHASE_EXPANSION", True):
                        obs_i.extend(self.dic_traffic_env_conf['PHASE'][states[i][feature][0]])
                    else:
                        obs_i.extend(states[i][feature])
                else:
                    values = np.asarray(states[i][feature], dtype=np.float32)
                    if self.use_phase_reindex_obs and values.size == 12:
                        values = self._phase_reindex_12(values, phase_idx)

                    if feature in lane_scale_features and self.obs_lane_num_scale != 1.0:
                        values = values / self.obs_lane_num_scale
                    elif feature == pressure_feature and self.obs_pressure_scale != 1.0:
                        values = values / self.obs_pressure_scale

                    obs_i.extend(values.tolist())
            obs_list.append(obs_i)
            adj_list.append(states[i]["adjacency_matrix"])
        
        obs = torch.tensor([obs_list], dtype=torch.float32, device=self.device)
        
        adj_np = np.array([adj_list])
        adj_sorted = np.sort(adj_np, axis=-1)
        adj_onehot = np.eye(self.num_agents)[adj_sorted]
        adj = torch.tensor(adj_onehot, dtype=torch.float32, device=self.device)
        
        return obs, adj

    @staticmethod
    def _phase_reindex_12(values: np.ndarray, phase_idx: int) -> np.ndarray:
        """
        CityLight-style phase reindex proxy on 12-dim lane-like features.
        Original approach order in this project is [W, E, N, S], each with 3 lanes.
        If current phase is vertical (2/4), reorder to [N, S, W, E] so the
        active axis is placed first.
        """
        if values.shape[0] != 12:
            return values
        if phase_idx in (2, 4):
            groups = [values[0:3], values[3:6], values[6:9], values[9:12]]
            return np.concatenate([groups[2], groups[3], groups[0], groups[1]], axis=0)
        return values
    
    def choose_action(self, count: int, states: List[Dict]) -> np.ndarray:
        obs, adj = self.convert_state_to_tensor(states)
        with torch.no_grad():
            actions, _, _, _ = self.network.get_action(obs, adj, deterministic=False)
        return actions.cpu().numpy()[0]
    
    def save_network(self, file_name: str):
        path = os.path.join(self.dic_path["PATH_TO_MODEL"], f"{file_name}.pt")
        torch.save(self.network.state_dict(), path)
    
    def load_network(self, file_name: str):
        path = os.path.join(self.dic_path["PATH_TO_MODEL"], f"{file_name}.pt")
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=self.device)
                self.network.load_state_dict(state_dict)
                print(f"Loaded PPO-CoLight network from {path}")
            except Exception as e:
                print(f"Error loading network: {e}")
        else:
            print(f"No checkpoint found at {path}, starting fresh")
