"""
MHQ-CoSLight agent.
observations: [lane_num_vehicle, cur_phase]
reward: -queue_length
"""
import json
import heapq
import numpy as np
import os
import shutil
import tempfile
from .agent import Agent
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    Attention,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    Lambda,
    Layer,
    LayerNormalization,
    Multiply,
    MultiHeadAttention,
    Reshape,
    LSTM,
)
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.utils import to_categorical


def build_memory():
    return []


class MatMulLayer(Layer):
    def call(self, x):
        return tf.matmul(x[0], x[1])

    def get_config(self):
        return super(MatMulLayer, self).get_config()


class PermuteDimensionsLayer(Layer):
    def call(self, x):
        return K.permute_dimensions(x, (0, 1, 4, 2, 3))

    def get_config(self):
        return super(PermuteDimensionsLayer, self).get_config()


class SoftmaxMatMulLayer(Layer):
    def call(self, x):
        return K.softmax(tf.matmul(x[0], x[1], transpose_b=True))

    def get_config(self):
        return super(SoftmaxMatMulLayer, self).get_config()


class MeanMatMulLayer(Layer):
    def call(self, x):
        return K.mean(tf.matmul(x[0], x[1]), axis=2)

    def get_config(self):
        return super(MeanMatMulLayer, self).get_config()


class SliceLayer(Layer):
    def __init__(self, start, end, **kwargs):
        self.start = start
        self.end = end
        super(SliceLayer, self).__init__(**kwargs)

    def call(self, x):
        return x[:, :, self.start:self.end, :]

    def get_config(self):
        config = {'start': self.start, 'end': self.end}
        base_config = super(SliceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ZerosLayer(Layer):
    def __init__(self, num_agents, head, num_neighbors, **kwargs):
        self.num_agents = num_agents
        self.head = head
        self.num_neighbors = num_neighbors
        super(ZerosLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.zeros((tf.shape(x)[0], self.num_agents, self.head, self.num_neighbors))

    def get_config(self):
        config = {
            'num_agents': self.num_agents,
            'head': self.head,
            'num_neighbors': self.num_neighbors
        }
        base_config = super(ZerosLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class StackHeads(Layer):
    """Stack N head outputs along a new axis. Serializable for model save/load."""
    def __init__(self, **kwargs):
        super(StackHeads, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs: list of [B, Agents, A] tensors
        return tf.stack(inputs, axis=2)  # [B, Agents, N, A]

    def get_config(self):
        return super(StackHeads, self).get_config()


class NoisyDense(Layer):
    """Factorized NoisyNet dense layer for exploration-aware MLPs."""
    def __init__(self, units, activation=None, sigma_init=0.017, use_bias=True, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.sigma_init = float(sigma_init)
        self.use_bias = bool(use_bias)

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        mu_range = 1.0 / np.sqrt(last_dim)
        self.kernel_mu = self.add_weight(
            name="kernel_mu",
            shape=(last_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
            trainable=True,
        )
        self.kernel_sigma = self.add_weight(
            name="kernel_sigma",
            shape=(last_dim, self.units),
            initializer=tf.keras.initializers.Constant(self.sigma_init / np.sqrt(last_dim)),
            trainable=True,
        )
        if self.use_bias:
            self.bias_mu = self.add_weight(
                name="bias_mu",
                shape=(self.units,),
                initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
                trainable=True,
            )
            self.bias_sigma = self.add_weight(
                name="bias_sigma",
                shape=(self.units,),
                initializer=tf.keras.initializers.Constant(self.sigma_init / np.sqrt(self.units)),
                trainable=True,
            )
        else:
            self.bias_mu = None
            self.bias_sigma = None
        super(NoisyDense, self).build(input_shape)

    @staticmethod
    def _f(x):
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def call(self, inputs, training=None):
        eps_in = tf.random.normal((tf.shape(inputs)[-1],), dtype=inputs.dtype)
        eps_out = tf.random.normal((self.units,), dtype=inputs.dtype)
        eps_in = self._f(eps_in)
        eps_out = self._f(eps_out)
        kernel_noise = tf.expand_dims(eps_in, -1) * tf.expand_dims(eps_out, 0)
        kernel = self.kernel_mu + self.kernel_sigma * kernel_noise
        outputs = tf.tensordot(inputs, kernel, axes=[[-1], [0]])
        if self.use_bias:
            bias = self.bias_mu + self.bias_sigma * eps_out
            outputs = outputs + bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation),
            "sigma_init": self.sigma_init,
            "use_bias": self.use_bias,
        }
        base_config = super(NoisyDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchRenorm(Layer):
    """
    Lightweight Batch Renormalization with warm-up for off-policy critic training.

    This mirrors the local CrossQ prototype closely enough for a safe REDQ-side
    migration while keeping serialization simple inside the existing Keras model.
    """

    def __init__(
        self,
        momentum=0.99,
        epsilon=1e-3,
        rmax=3.0,
        dmax=5.0,
        warmup_steps=100000,
        center=True,
        scale=True,
        **kwargs
    ):
        super(BatchRenorm, self).__init__(**kwargs)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.rmax = float(rmax)
        self.dmax = float(dmax)
        self.warmup_steps = int(warmup_steps)
        self.center = bool(center)
        self.scale = bool(scale)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer="ones",
                trainable=True,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=(dim,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(dim,),
            initializer="zeros",
            trainable=False,
        )
        self.moving_var = self.add_weight(
            name="moving_var",
            shape=(dim,),
            initializer="ones",
            trainable=False,
        )
        self.steps = self.add_weight(
            name="steps",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype=tf.int64,
        )
        super(BatchRenorm, self).build(input_shape)

    def _training_call(self, inputs):
        reduction_axes = list(range(len(inputs.shape) - 1))
        batch_mean, batch_var = tf.nn.moments(inputs, axes=reduction_axes, keepdims=False)

        moving_std = tf.sqrt(self.moving_var + self.epsilon)
        batch_std = tf.sqrt(batch_var + self.epsilon)
        r = tf.stop_gradient(batch_std / moving_std)
        r = tf.clip_by_value(r, 1.0 / self.rmax, self.rmax)
        d = tf.stop_gradient((batch_mean - self.moving_mean) / moving_std)
        d = tf.clip_by_value(d, -self.dmax, self.dmax)

        normalized = (inputs - batch_mean) / batch_std
        renormed = normalized * r + d
        warmed_up = tf.cast(tf.greater_equal(self.steps, self.warmup_steps), normalized.dtype)
        normalized = warmed_up * renormed + (1.0 - warmed_up) * normalized

        self.moving_mean.assign(self.momentum * self.moving_mean + (1.0 - self.momentum) * batch_mean)
        self.moving_var.assign(self.momentum * self.moving_var + (1.0 - self.momentum) * batch_var)
        self.steps.assign_add(tf.constant(1, dtype=tf.int64))
        return self._affine(normalized)

    def _inference_call(self, inputs):
        normalized = (inputs - self.moving_mean) / tf.sqrt(self.moving_var + self.epsilon)
        return self._affine(normalized)

    def _affine(self, inputs):
        outputs = inputs
        if self.scale:
            outputs = outputs * self.gamma
        if self.center:
            outputs = outputs + self.beta
        return outputs

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        if isinstance(training, bool):
            return self._training_call(inputs) if training else self._inference_call(inputs)
        return tf.__internal__.smart_cond.smart_cond(
            training,
            lambda: self._training_call(inputs),
            lambda: self._inference_call(inputs),
        )

    def get_config(self):
        config = {
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "rmax": self.rmax,
            "dmax": self.dmax,
            "warmup_steps": self.warmup_steps,
            "center": self.center,
            "scale": self.scale,
        }
        base_config = super(BatchRenorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FRAPRelationTile(Layer):
    """
    Tile a fixed FRAP phase-pair relation matrix to [B, Agents, P, P, R].
    Serializable replacement for Lambda+captured tensor constants.
    """
    def __init__(self, rel_matrix, **kwargs):
        super(FRAPRelationTile, self).__init__(**kwargs)
        self.rel_matrix = np.array(rel_matrix, dtype=np.float32)

    def call(self, x):
        r = tf.convert_to_tensor(self.rel_matrix, dtype=tf.float32)  # [P, P, R]
        r = tf.expand_dims(tf.expand_dims(r, axis=0), axis=0)        # [1,1,P,P,R]
        return tf.tile(r, [tf.shape(x)[0], tf.shape(x)[1], 1, 1, 1])

    def get_config(self):
        cfg = super(FRAPRelationTile, self).get_config()
        cfg.update({"rel_matrix": self.rel_matrix.tolist()})
        return cfg


class FRAPBinaryRelationTile(Layer):
    """
    Tile fixed binary FRAP relation matrix to [B, Agents, 8, 7].
    """
    def __init__(self, rel_matrix, **kwargs):
        super(FRAPBinaryRelationTile, self).__init__(**kwargs)
        self.rel_matrix = np.array(rel_matrix, dtype=np.int32)

    def call(self, x):
        r = tf.convert_to_tensor(self.rel_matrix, dtype=tf.int32)  # [8,7]
        r = tf.expand_dims(tf.expand_dims(r, axis=0), axis=0)      # [1,1,8,7]
        return tf.tile(r, [tf.shape(x)[0], tf.shape(x)[1], 1, 1])

    def get_config(self):
        cfg = super(FRAPBinaryRelationTile, self).get_config()
        cfg.update({"rel_matrix": self.rel_matrix.tolist()})
        return cfg


class CoSDynamicAdjacency(Layer):
    """
    Build dynamic collaborator adjacency from CoS logits/probabilities.
    Output shape: [B, N, K, N], where K includes self when include_self=True.
    """
    def __init__(
        self,
        num_agents,
        total_k,
        include_self=True,
        adj_mode="tiled_sparse",
        slot_min_prob=0.0,
        slot_budget_tau=0.05,
        explore_mode="none",
        explore_prob=0.0,
        gumbel_tau=1.0,
        gumbel_scale=1.0,
        explore_infer=False,
        **kwargs
    ):
        super(CoSDynamicAdjacency, self).__init__(**kwargs)
        self.num_agents = int(num_agents)
        self.total_k = int(total_k)
        self.include_self = bool(include_self)
        self.adj_mode = str(adj_mode or "tiled_sparse").lower()
        self.slot_min_prob = float(slot_min_prob)
        self.slot_budget_tau = float(slot_budget_tau)
        self.explore_mode = str(explore_mode or "none").lower()
        self.explore_prob = float(explore_prob)
        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_scale = float(gumbel_scale)
        self.explore_infer = bool(explore_infer)

    def call(self, inputs, training=None):
        # inputs: scores [B, N, N] or [scores, candidate_adj]
        if isinstance(inputs, (list, tuple)):
            scores = inputs[0]
            candidate_adj = inputs[1] if len(inputs) > 1 else None
        else:
            scores = inputs
            candidate_adj = None
        scores = tf.cast(scores, tf.float32)
        if candidate_adj is not None:
            cand_mask = tf.reduce_sum(candidate_adj, axis=2)  # [B,N,N]
            cand_mask = tf.cast(cand_mask > 0, scores.dtype)
            batch = tf.shape(scores)[0]
            eye = tf.tile(
                tf.expand_dims(tf.eye(self.num_agents, dtype=scores.dtype), axis=0),
                [batch, 1, 1],
            )
            cand_mask = tf.maximum(cand_mask, eye)
            neg_large = tf.constant(-1e9, dtype=scores.dtype)
            scores = tf.where(cand_mask > 0, scores, neg_large)
        probs = tf.nn.softmax(scores, axis=-1)
        dtype = probs.dtype
        batch = tf.shape(probs)[0]
        n = self.num_agents

        eye = tf.eye(n, dtype=dtype)  # [N, N]
        eye_b = tf.tile(tf.reshape(eye, [1, n, n]), [batch, 1, 1])  # [B, N, N]

        # Clamp K to valid range.
        max_k = n if not self.include_self else max(n - 1, 0)
        other_k = self.total_k - (1 if self.include_self else 0)
        other_k = max(0, min(other_k, max_k))

        if self.include_self:
            probs_others = probs * (1.0 - eye_b)
        else:
            probs_others = probs

        if other_k > 0:
            # Deterministic: top-k on probabilities (current behavior).
            det_topk = tf.math.top_k(probs_others, k=other_k, sorted=False)
            indices = det_topk.indices

            # Optional: stochastic, without-replacement sampling via Gumbel-TopK on log-probs.
            # This matches CoSLight's intent (exploration in collaborator selection) while
            # keeping a deterministic default (explore_prob=0).
            can_sample = (
                self.explore_mode in ("gumbel_topk", "gumbel", "sample", "sampling")
                and (self.explore_prob > 0.0)
            )
            if can_sample:
                eps = tf.constant(1e-20, dtype=dtype)
                log_probs = tf.math.log(probs_others + eps)
                u = tf.random.uniform(tf.shape(log_probs), minval=0.0, maxval=1.0, dtype=dtype)
                g = -tf.math.log(-tf.math.log(u + eps) + eps)  # Gumbel(0,1)
                tau = tf.constant(max(self.gumbel_tau, 1e-6), dtype=dtype)
                noisy = (log_probs / tau) + (tf.constant(self.gumbel_scale, dtype=dtype) * g)
                samp_topk = tf.math.top_k(noisy, k=other_k, sorted=False)

                # Decide per (batch, agent-row) whether to sample.
                # Default: only during training; can be forced during inference via explore_infer.
                do_phase = (training is True) or self.explore_infer
                if do_phase:
                    row_u = tf.random.uniform([batch, n, 1], minval=0.0, maxval=1.0, dtype=dtype)
                    use_sample = row_u < tf.constant(self.explore_prob, dtype=dtype)
                    use_sample = tf.tile(use_sample, [1, 1, other_k])  # [B,N,K]
                    indices = tf.where(use_sample, samp_topk.indices, indices)

            if self.adj_mode in ("topk_slots", "slots", "topk"):
                # Build K distinct neighbor slots: each slot is a (scaled) one-hot vector.
                # This is more faithful to Top-K collaborator selection and enables
                # adaptive effective-K by thresholding small probabilities.
                topk_p = tf.gather(probs_others, indices, batch_dims=2)  # [B,N,K]

                if self.slot_min_prob > 0.0:
                    thr = tf.constant(self.slot_min_prob, dtype=dtype)
                    tau = tf.constant(max(self.slot_budget_tau, 1e-6), dtype=dtype)
                    keep = tf.sigmoid((topk_p - thr) / tau)  # [B,N,K]
                    topk_p = topk_p * keep

                onehots = tf.one_hot(indices, depth=n, dtype=dtype)  # [B,N,K,N]
                other_rows = onehots * tf.expand_dims(topk_p, axis=-1)  # [B,N,K,N]
            else:
                # Legacy behavior: build a sparse distribution over selected set, then tile.
                mask = tf.reduce_sum(tf.one_hot(indices, depth=n, dtype=dtype), axis=2)  # [B,N,N]
                sparse = probs_others * mask
                sparse = sparse / (tf.reduce_sum(sparse, axis=-1, keepdims=True) + 1e-8)
                other_rows = tf.tile(tf.expand_dims(sparse, axis=2), [1, 1, other_k, 1])  # [B,N,other_k,N]
        else:
            other_rows = tf.zeros([batch, n, 0, n], dtype=dtype)

        if self.include_self:
            self_row = tf.tile(tf.reshape(eye, [1, n, 1, n]), [batch, 1, 1, 1])  # [B,N,1,N]
            return tf.concat([self_row, other_rows], axis=2)
        return other_rows

    def get_config(self):
        config = {
            "num_agents": self.num_agents,
            "total_k": self.total_k,
            "include_self": self.include_self,
            "adj_mode": self.adj_mode,
            "slot_min_prob": self.slot_min_prob,
            "slot_budget_tau": self.slot_budget_tau,
            "explore_mode": self.explore_mode,
            "explore_prob": self.explore_prob,
            "gumbel_tau": self.gumbel_tau,
            "gumbel_scale": self.gumbel_scale,
            "explore_infer": self.explore_infer,
        }
        base = super(CoSDynamicAdjacency, self).get_config()
        return dict(list(base.items()) + list(config.items()))


class MHQCoSLightAgent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        super(MHQCoSLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.cos_enabled = bool(dic_agent_conf.get("COS_ENABLED", True))
        self.cos_total_k = int(dic_agent_conf.get("COS_TOTAL_K", dic_traffic_env_conf.get("TOP_K_ADJACENCY", 5)))
        self.use_dynamic_collab_full = bool(dic_agent_conf.get("USE_DYNAMIC_COLLAB_FULL", False))
        self.neighbor_select_enabled = False
        self.neighbor_candidate_hop = int(dic_agent_conf.get("NEIGHBOR_CANDIDATE_HOP", 2))
        self.neighbor_topk = int(dic_agent_conf.get("NEIGHBOR_TOPK", 5))
        self.neighbor_gate_type = str(dic_agent_conf.get("NEIGHBOR_GATE_TYPE", "soft") or "soft").lower()
        self.neighbor_gate_threshold = float(dic_agent_conf.get("NEIGHBOR_GATE_THRESHOLD", 0.1))
        self.neighbor_gate_temp = float(dic_agent_conf.get("NEIGHBOR_GATE_TEMP", 0.05))
        self.use_topo_feature = bool(dic_agent_conf.get("USE_TOPO_FEATURE", True))
        self.use_delay_feature = bool(dic_agent_conf.get("USE_DELAY_FEATURE", True))
        self.use_same_corridor_feature = bool(dic_agent_conf.get("USE_SAME_CORRIDOR_FEATURE", False))
        self.delay_use_distance_only = bool(dic_agent_conf.get("DELAY_USE_DISTANCE_ONLY", False))
        self.relation_hidden_dim = int(dic_agent_conf.get("RELATION_HIDDEN_DIM", 32))
        self.neighbor_state_rel_mode = str(dic_agent_conf.get("NEIGHBOR_STATE_REL_MODE", "diff_only") or "diff_only").lower()
        self.use_neighbor_h_mean_concat = bool(dic_agent_conf.get("USE_NEIGHBOR_H_MEAN_CONCAT", False))
        self.use_delay_msg_mean = bool(dic_agent_conf.get("USE_DELAY_MSG_MEAN", False))
        self.use_delay_rel_msg_mean = bool(dic_agent_conf.get("USE_DELAY_REL_MSG_MEAN", False))
        if sum(
            bool(x)
            for x in (
                self.use_neighbor_h_mean_concat,
                self.use_delay_msg_mean,
                self.use_delay_rel_msg_mean,
            )
        ) > 1:
            raise ValueError(
                "USE_NEIGHBOR_H_MEAN_CONCAT, USE_DELAY_MSG_MEAN and USE_DELAY_REL_MSG_MEAN cannot be enabled together."
            )
        self.delay_msg_hidden_dim = int(dic_agent_conf.get("DELAY_MSG_HIDDEN_DIM", dic_agent_conf.get("CRITIC_HIDDEN_DIM", 32)))
        self.delay_msg_tau_norm_mode = str(dic_agent_conf.get("DELAY_MSG_TAU_NORM_MODE", "min_action_time") or "min_action_time").lower()
        self.delay_msg_delta_reduce = str(dic_agent_conf.get("DELAY_MSG_DELTA_REDUCE", "mean") or "mean").lower()
        self.critic_activation = str(dic_agent_conf.get("CRITIC_ACTIVATION", "relu") or "relu").lower()
        if self.critic_activation not in ("relu", "sigmoid", "linear"):
            raise ValueError("Unsupported CRITIC_ACTIVATION: {}".format(self.critic_activation))
        self.static_relation_cache = None
        self.static_delay_msg_cache = self._build_static_delay_msg_cache()
        if (self.use_delay_msg_mean or self.use_delay_rel_msg_mean) and self.static_delay_msg_cache is None:
            raise ValueError("StaticDelay message mean requires a valid tau cache from roadnet.")
        self.dynamic_collab_pair_dim = int(dic_agent_conf.get("DYNAMIC_COLLAB_PAIR_DIM", 32))
        self.dynamic_collab_need_bias = float(dic_agent_conf.get("DYNAMIC_COLLAB_NEED_BIAS", 2.0))
        self.cos_include_self = bool(dic_agent_conf.get("COS_INCLUDE_SELF", True))
        self.cos_use_input_candidate_mask = bool(dic_agent_conf.get("COS_USE_INPUT_CANDIDATE_MASK", False))
        self.cos_beta_diag = float(dic_agent_conf.get("COS_BETA_DIAG", 0.0))
        self.cos_gamma_sym = float(dic_agent_conf.get("COS_GAMMA_SYM", 0.0))
        self.cos_entropy_coef = float(dic_agent_conf.get("COS_ENTROPY_COEF", 0.0))
        self.cos_temporal_smooth_coef = float(dic_agent_conf.get("COS_TEMPORAL_SMOOTH_COEF", 0.0))
        self.cos_budget_coef = float(dic_agent_conf.get("COS_BUDGET_COEF", 0.0))
        self.cos_budget_thr = float(dic_agent_conf.get("COS_BUDGET_THR", 0.0))
        self.cos_budget_tau = float(dic_agent_conf.get("COS_BUDGET_TAU", 0.05))
        self.use_intersection_pos_enc = bool(dic_agent_conf.get("USE_INTERSECTION_POS_ENC", False))
        self.intersection_pos_dim = int(dic_agent_conf.get("INTERSECTION_POS_DIM", 16))
        self.cos_adj_mode = str(dic_agent_conf.get("COS_ADJ_MODE", "tiled_sparse") or "tiled_sparse").lower()
        self.cos_slot_min_prob = float(dic_agent_conf.get("COS_SLOT_MIN_PROB", 0.0))
        # CoS collaborator selection exploration (default OFF).
        self.cos_explore_mode = str(dic_agent_conf.get("COS_EXPLORE_MODE", "none") or "none").lower()
        self.cos_explore_prob = float(dic_agent_conf.get("COS_EXPLORE_PROB", 0.0))
        self.cos_gumbel_tau = float(dic_agent_conf.get("COS_GUMBEL_TAU", 1.0))
        self.cos_gumbel_scale = float(dic_agent_conf.get("COS_GUMBEL_SCALE", 1.0))
        self.cos_explore_infer = bool(dic_agent_conf.get("COS_EXPLORE_INFER", False))
        # Separate candidate-neighbor slots from final collaborator budget K.
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)
        self.cos_select_k = min(self.cos_total_k, self.num_agents)

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()
        self.feature_slices = self._feature_slice_bounds()
        self.memory = build_memory()
        self.tsa_enabled = False
        self.tsa_gaussian_std = 0.0
        self.tsa_mask_prob = 0.0
        self.tsa_scale_low = 1.0
        self.tsa_scale_high = 1.0
        self.tsa_apply_to_next_state = True
        self.tsa_dim_mask = np.ones((1, 1, self.len_feature), dtype=np.float32)
        self.use_q_consistency_aux = False
        self.q_consistency_weight = 0.0
        self.use_auxiliary_head = bool(dic_agent_conf.get("USE_AUXILIARY_HEAD", False))
        self.auxiliary_task = str(dic_agent_conf.get("AUXILIARY_TASK", "none") or "none").lower()
        self.auxiliary_weight = float(dic_agent_conf.get("AUXILIARY_WEIGHT", 0.0))
        self.auxiliary_ema_tau = float(dic_agent_conf.get("AUXILIARY_EMA_TAU", 0.995))
        self.use_isr = False
        self.use_latent_transition_ssl = bool(
            self.use_auxiliary_head and self.auxiliary_task == "latent_transition" and self.auxiliary_weight > 0.0
        )
        self.ssl_latent_dim = 0
        self.ssl_online_encoders = []
        self.ssl_target_encoders = []
        self.ssl_transition_models = []
        self.ssl_action_onehot = None
        self.ssl_next_inputs = None
        self.ssl_sample_weight = None
        self.Y_isr = None
        self.use_feature_group_gate = bool(dic_agent_conf.get("USE_FEATURE_GROUP_GATE", False))
        self.use_feature_group_concat = bool(dic_agent_conf.get("USE_FEATURE_GROUP_CONCAT", False))
        self.feature_group_hidden_dim = int(dic_agent_conf.get("FEATURE_GROUP_HIDDEN_DIM", 16))
        self.use_per = bool(dic_agent_conf.get("USE_PER", False))
        self.per_alpha = float(dic_agent_conf.get("PER_ALPHA", 0.6))
        self.per_eps = float(dic_agent_conf.get("PER_EPS", 1e-3))
        self.per_uniform_mix = float(dic_agent_conf.get("PER_UNIFORM_MIX", 0.1))
        self.per_uniform_mix = float(np.clip(self.per_uniform_mix, 0.0, 1.0))
        self.per_priority_clip = 0.0
        self.per_warmup_rounds = 0
        self.per_is_weights = None
        self.action_gaussian_std = float(dic_agent_conf.get("ACTION_GAUSSIAN_STD", 0.0))
        self.action_gaussian_clip = float(dic_agent_conf.get("ACTION_GAUSSIAN_CLIP", 0.0))
        self.use_noisy_net = bool(dic_agent_conf.get("USE_NOISY_NET", False))
        self.noisy_sigma_init = float(dic_agent_conf.get("NOISY_SIGMA_INIT", 0.017))

        self.use_multihead = dic_agent_conf.get("USE_MULTIHEAD_Q", False)
        self.head_n = dic_agent_conf.get("HEAD_N", 5)
        self.head_agg = str(dic_agent_conf.get("HEAD_AGG", "mean")).lower()
        self.head_debug = dic_agent_conf.get("HEAD_DEBUG", False)
        self.use_ucb_action = bool(dic_agent_conf.get("USE_UCB_ACTION", False))
        self.ucb_lambda = float(dic_agent_conf.get("UCB_LAMBDA", 0.2))
        self.ucb_decay = float(dic_agent_conf.get("UCB_DECAY", 1.0))
        self.ucb_min = float(dic_agent_conf.get("UCB_MIN", 0.0))
        self.use_head_bootstrap = bool(dic_agent_conf.get("USE_HEAD_BOOTSTRAP", False))
        self.head_bootstrap_p = float(np.clip(dic_agent_conf.get("HEAD_BOOTSTRAP_P", 0.8), 0.0, 1.0))
        self.use_redq = dic_agent_conf.get("USE_REDQ", False)
        self.redq_m = int(dic_agent_conf.get("REDQ_M", 2))
        self.redq_lambda = float(dic_agent_conf.get("REDQ_LAMBDA", 1.0))
        self.redq_utd = max(1, int(dic_agent_conf.get("REDQ_UTD", 1)))
        # True REDQ mode: use independent Q-ensemble (not shared-trunk multi-head).
        self.true_redq_mode = bool(dic_agent_conf.get("TRUE_REDQ_MODE", False))
        self.use_true_redq_ensemble = bool(self.use_redq and self.true_redq_mode)
        self.redq_n = max(2, int(dic_agent_conf.get("REDQ_N", self.head_n)))
        # RELight-style discrete ensemble action selection.
        self.relight_action_vote = bool(dic_agent_conf.get("RELIGHT_ACTION_VOTE", False))
        # Rainbow-lite algorithm improvements.
        self.use_dueling = bool(dic_agent_conf.get("USE_DUELING", False))
        self.use_double_dqn = bool(dic_agent_conf.get("USE_DOUBLE_DQN", False))
        self.nstep = max(1, int(dic_agent_conf.get("NSTEP", 1)))
        self.critic_use_layer_norm = bool(dic_agent_conf.get("CRITIC_USE_LAYER_NORM", False))
        self.critic_dropout_rate = float(dic_agent_conf.get("CRITIC_DROPOUT_RATE", 0.0))
        self.critic_hidden_dim = max(1, int(dic_agent_conf.get("CRITIC_HIDDEN_DIM", 32)))
        self.critic_num_layers = max(1, int(dic_agent_conf.get("CRITIC_NUM_LAYERS", 2)))
        self.droq_mode = bool(dic_agent_conf.get("DROQ_MODE", False))
        self.crossq_safe_mode = bool(dic_agent_conf.get("CROSSQ_SAFE_MODE", False))
        self.crossq_bn_mode = str(dic_agent_conf.get("CROSSQ_BN_MODE", "brn") or "brn").lower()
        self.crossq_use_batch_norm = bool(dic_agent_conf.get("CROSSQ_USE_BATCH_NORM", False))
        self.crossq_batch_norm_momentum = float(dic_agent_conf.get("CROSSQ_BATCH_NORM_MOMENTUM", 0.99))
        self.crossq_brn_warmup_steps = int(dic_agent_conf.get("CROSSQ_BRN_WARMUP_STEPS", 100000))
        self.crossq_use_live_bnstats_for_target = bool(dic_agent_conf.get("CROSSQ_USE_LIVE_BNSTATS_FOR_TARGET", False))
        self.crossq_joint_forward = bool(dic_agent_conf.get("CROSSQ_JOINT_FORWARD", False))
        self.crossq_custom_train_step = bool(dic_agent_conf.get("CROSSQ_CUSTOM_TRAIN_STEP", False))
        self.crossq_keep_target_net = bool(dic_agent_conf.get("CROSSQ_KEEP_TARGET_NET", True))
        self.use_adaptive_pressure_phase_head = bool(
            dic_agent_conf.get("USE_ADAPTIVE_PRESSURE_PHASE_HEAD", False)
        )
        self.use_light_phase_relation = bool(dic_agent_conf.get("USE_LIGHT_PHASE_RELATION", False))
        self.use_light_temporal_delta = bool(dic_agent_conf.get("USE_LIGHT_TEMPORAL_DELTA", False))
        self.apl_move_hidden_dim = int(dic_agent_conf.get("APL_MOVE_HIDDEN_DIM", 16))
        self.apl_phase_hidden_dim = int(dic_agent_conf.get("APL_PHASE_HIDDEN_DIM", 16))
        self.apl_rel_dim = int(dic_agent_conf.get("APL_REL_DIM", 16))
        self.apl_temporal_delta_weight = float(dic_agent_conf.get("APL_TEMPORAL_DELTA_WEIGHT", 0.25))
        if self.use_true_redq_ensemble:
            # In true REDQ mode, HEAD_N is interpreted as ensemble size N.
            self.use_multihead = False
            self.head_n = self.redq_n
            self.redq_m = max(1, min(self.redq_m, self.redq_n))
        else:
            self.redq_m = max(1, min(self.redq_m, self.head_n))

        # CityLight-inspired: competitive neighbor aggregation
        self.use_competitive_agg = dic_agent_conf.get("USE_COMPETITIVE_AGG", False)
        self.use_gat_agg = bool(dic_agent_conf.get("USE_GAT_AGG", True))
        self.use_mlp_neighbor_agg = bool(dic_agent_conf.get("USE_MLP_NEIGHBOR_AGG", False))
        # Optional CoSLight-style Transformer encoder (default OFF to keep behavior unchanged).
        self.use_transformer_encoder = bool(dic_agent_conf.get("USE_TRANSFORMER_ENCODER", False))
        self.trans_dim = int(dic_agent_conf.get("TRANS_DIM", 0))
        self.trans_heads = int(dic_agent_conf.get("TRANS_HEADS", 4))
        self.trans_layers = int(dic_agent_conf.get("TRANS_LAYERS", 2))
        self.trans_ffn_dim = int(dic_agent_conf.get("TRANS_FFN_DIM", 128))
        self.trans_dropout = float(dic_agent_conf.get("TRANS_DROPOUT", 0.1))
        self.trans_use_cos_mask = bool(dic_agent_conf.get("TRANS_USE_COS_MASK", True))
        self.trans_prenorm = bool(dic_agent_conf.get("TRANS_PRENORM", True))
        self.use_block_attn_res = bool(dic_agent_conf.get("USE_BLOCK_ATTN_RES", False))
        self.use_frap_phase_compete = False
        self.use_frap_strict = False
        self.use_official_redq_update = False
        self.cos_prob_model = None
        
        if self.use_true_redq_ensemble:
            print(
                "[True-REDQ] enabled, N={}, M={}, λ={}, UTD={} "
                "(independent critics; no shared-trunk multi-head)".format(
                    self.redq_n, self.redq_m, self.redq_lambda, self.redq_utd
                )
            )
        if self.use_per:
            print(
                "[PER] enabled, alpha={}, eps={}, uniform_mix={}".format(
                    self.per_alpha, self.per_eps, self.per_uniform_mix
                )
            )
        if self.relight_action_vote:
            print("[RELight] vote action enabled (majority vote across ensemble critics)")
        if self.action_gaussian_std > 0:
            print(
                "[ActionGaussian] enabled, std={}, clip={}".format(
                    self.action_gaussian_std, self.action_gaussian_clip
                )
            )
        if self.use_noisy_net:
            print("[NoisyNet] enabled, sigma_init={}".format(self.noisy_sigma_init))
        if self.use_multihead:
            print("[MultiHead] enabled, N={}, AGG={}".format(
                self.head_n, self.head_agg))
            if self.head_agg not in ("mean", "trimmed_mean"):
                raise ValueError("HEAD_AGG must be one of: mean, trimmed_mean")
            if self.use_ucb_action:
                print("[MultiHead-UCB] enabled, lambda={}, decay={}, min={}".format(
                    self.ucb_lambda, self.ucb_decay, self.ucb_min))
            if self.use_head_bootstrap:
                print("[MultiHead-Bootstrap] enabled, p={}".format(self.head_bootstrap_p))
            if self.use_redq:
                print("[REDQ] enabled, M={}, λ={}, UTD={} (Q_mix = (1-λ)*mean + λ*min_sub)".format(
                    self.redq_m, self.redq_lambda, self.redq_utd))
                if self.true_redq_mode:
                    print("[REDQ] true_redq_mode=True (legacy multi-head action mixing path)")
        if self.use_dueling:
            print("[Dueling] DQN enabled: Q = V(s) + A(s,a) - mean(A)")
        if self.use_double_dqn:
            print("[Double] DQN enabled: online net selects action, target net evaluates")
        if self.critic_use_layer_norm or self.critic_dropout_rate > 0:
            print(
                "[CriticReg] ln={}, dropout={}, hidden_dim={}, layers={}, droq_mode={}".format(
                    self.critic_use_layer_norm,
                    self.critic_dropout_rate,
                    self.critic_hidden_dim,
                    self.critic_num_layers,
                    self.droq_mode,
                )
            )
        if self.crossq_safe_mode:
            print(
                "[CrossQ-Safe] enabled, bn_mode={}, use_bn={}, joint_forward={}, custom_train_step={}, keep_target={}, live_target_stats={}".format(
                    self.crossq_bn_mode,
                    self.crossq_use_batch_norm,
                    self.crossq_joint_forward,
                    self.crossq_custom_train_step,
                    self.crossq_keep_target_net,
                    self.crossq_use_live_bnstats_for_target,
                )
            )
        if self.use_adaptive_pressure_phase_head:
            print(
                "[APLight] adaptive pressure head enabled, phase_relation={}, temporal_delta={}".format(
                    self.use_light_phase_relation,
                    self.use_light_temporal_delta,
                )
            )
        if self.nstep > 1:
            print("[N-step] n={}, bootstrap γ^n={:.4f}".format(self.nstep, self.dic_agent_conf["GAMMA"] ** self.nstep))
        if self.use_competitive_agg:
            print("[CompetitiveAgg] enabled, splitting neighbors into 2 competing groups")
        if (not self.use_transformer_encoder) and (not self.use_gat_agg):
            print("[MLP-Only] enabled, bypassing neighbor attention aggregation")
            if self.use_mlp_neighbor_agg:
                print("[MLP-NeighborAgg] enabled, residual neighbor mean aggregation before action head")
            if self.cos_enabled:
                print(
                    "[DynamicCollab] enabled in MLP-only path, CoS adjacency with K={} feeds residual neighbor aggregation".format(
                        self.cos_select_k
                    )
                )
        if self.use_transformer_encoder:
            print(
                "[Transformer] enabled, dim={}, heads={}, layers={}, ffn_dim={}, "
                "dropout={}, use_mask={}, prenorm={}, block_attn_res={}".format(
                    self.trans_dim if self.trans_dim > 0 else self.CNN_layers[0][1],
                    self.trans_heads,
                    self.trans_layers,
                    self.trans_ffn_dim,
                    self.trans_dropout,
                    self.trans_use_cos_mask,
                    self.trans_prenorm,
                    self.use_block_attn_res,
                )
            )
        if self.use_frap_phase_compete:
            print("[FRAP] phase-competition encoder enabled (MHQCoSLight lightweight adaptation)")
            if self.use_frap_strict:
                print("[FRAP] strict pair-conv mode enabled (CoSLight-style structure)")
        if self.tsa_enabled:
            active_ratio = float(np.mean(self.tsa_dim_mask)) if self.tsa_dim_mask.size > 0 else 0.0
            print(
                "[TSA] enabled, gaussian_std={}, mask_prob={}, scale_range=[{}, {}], "
                "apply_to_next_state={}, active_dim_ratio={:.3f}".format(
                    self.tsa_gaussian_std,
                    self.tsa_mask_prob,
                    self.tsa_scale_low,
                    self.tsa_scale_high,
                    self.tsa_apply_to_next_state,
                    active_ratio,
                )
            )
        if self.use_q_consistency_aux:
            print(
                "[Q-Consistency] enabled, weight={}, TSA-backed augmentation".format(
                    self.q_consistency_weight
                )
            )
        if self.use_auxiliary_head:
            print(
                "[AuxHead] enabled, task={}, weight={}".format(
                    self.auxiliary_task, self.auxiliary_weight
                )
            )
            if self.use_latent_transition_ssl:
                print(
                    "[AuxSSL] latent transition enabled, ema_tau={}".format(
                        self.auxiliary_ema_tau
                    )
                )
        if self.cos_enabled:
            print("[CoS] enabled, K={}, include_self={}, beta_diag={}, gamma_sym={}, ent_coef={}".format(
                self.cos_select_k,
                self.cos_include_self,
                self.cos_beta_diag,
                self.cos_gamma_sym,
                self.cos_entropy_coef,
            ))
            if self.cos_use_input_candidate_mask:
                print("[CoS] candidate mask from input adjacency enabled")
            if self.use_intersection_pos_enc:
                print("[PosEnc] enabled, dim={}".format(self.intersection_pos_dim))
            if self.cos_explore_mode != "none" and self.cos_explore_prob > 0.0:
                print(
                    "[CoS-Explore] mode={}, prob={}, tau={}, scale={}, infer={}".format(
                        self.cos_explore_mode,
                        self.cos_explore_prob,
                        self.cos_gumbel_tau,
                        self.cos_gumbel_scale,
                        self.cos_explore_infer,
                    )
                )
        if self.use_true_redq_ensemble:
            self._init_true_redq_ensemble(cnt_round, intersection_id)
        else:
            if cnt_round == 0:
                # initialization
                self.q_network = self.build_network()
                self._refresh_cos_prob_model()
                init_loaded = self._load_init_checkpoint_if_available()
                if (not init_loaded) and os.listdir(self.dic_path["PATH_TO_MODEL"]):
                    self.q_network.load_weights(
                        os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(intersection_id)),
                        by_name=True)
                if not init_loaded:
                    self.q_network_bar = self.build_network_from_copy(self.q_network)
            else:
                try:
                    self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
                    if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                        if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf[
                                    "UPDATE_Q_BAR_FREQ"], 0),
                                self.intersection_id))
                        else:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
                except:
                    print("fail to load network, current round: {0}".format(cnt_round))
                self._refresh_cos_prob_model()
            self._init_auxiliary_ssl_models()

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    @staticmethod
    def _custom_objects():
        return {
            "RepeatVector3D": RepeatVector3D,
            "StackHeads": StackHeads,
            "NoisyDense": NoisyDense,
            "BatchRenorm": BatchRenorm,
            "CoSDynamicAdjacency": CoSDynamicAdjacency,
            "FRAPRelationTile": FRAPRelationTile,
            "FRAPBinaryRelationTile": FRAPBinaryRelationTile,
            "MatMulLayer": MatMulLayer,
            "PermuteDimensionsLayer": PermuteDimensionsLayer,
            "SoftmaxMatMulLayer": SoftmaxMatMulLayer,
            "MeanMatMulLayer": MeanMatMulLayer,
            "SliceLayer": SliceLayer,
            "ZerosLayer": ZerosLayer
        }

    def _ensemble_file_name(self, file_name, q_idx):
        return "{}_q{}".format(file_name, q_idx)

    def _sample_redq_indices(self):
        if self.redq_m >= self.redq_n:
            return np.arange(self.redq_n, dtype=np.int32)
        return np.random.choice(self.redq_n, self.redq_m, replace=False)

    def _soft_update_true_redq_targets(self):
        """
        Paper-style Polyak soft update for true-REDQ ensemble:
        target <- (1 - tau) * target + tau * online
        """
        if (not self.use_true_redq_ensemble) or (not bool(self.dic_agent_conf.get("REDQ_SOFT_TARGET_UPDATE", False))):
            return
        tau = float(self.dic_agent_conf.get("REDQ_TAU", 0.005))
        tau = float(np.clip(tau, 0.0, 1.0))
        if tau <= 0.0:
            return
        for online_net, target_net in zip(self.q_ensemble, self.q_ensemble_bar):
            for target_var, online_var in zip(target_net.weights, online_net.weights):
                cast_online = tf.cast(online_var, target_var.dtype)
                if target_var.dtype.is_floating:
                    target_var.assign((1.0 - tau) * target_var + tau * cast_online)
                else:
                    target_var.assign(cast_online)
        self.q_network_bar = self.q_ensemble_bar[0]

    def _crossq_norm_layer(self, name):
        if not self.crossq_use_batch_norm:
            return None
        if self.crossq_bn_mode == "brn":
            return BatchRenorm(
                momentum=self.crossq_batch_norm_momentum,
                epsilon=1e-3,
                warmup_steps=self.crossq_brn_warmup_steps,
                name=name,
            )
        return BatchNormalization(
            momentum=self.crossq_batch_norm_momentum,
            epsilon=1e-3,
            name=name,
        )

    def _init_true_redq_ensemble(self, cnt_round, intersection_id):
        self.q_ensemble = []
        self.q_ensemble_bar = []
        use_soft_target = bool(self.dic_agent_conf.get("REDQ_SOFT_TARGET_UPDATE", False))
        if cnt_round == 0:
            # Fresh start: independent critics with independent initialization.
            self.q_ensemble = [self.build_network() for _ in range(self.redq_n)]
            self.q_ensemble_bar = [self.build_network_from_copy(net) for net in self.q_ensemble]
            self.q_network = self.q_ensemble[0]
            self.q_network_bar = self.q_ensemble_bar[0]
            self._load_init_checkpoint_if_available()
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
                if use_soft_target:
                    # Paper-style target update path starts each round from online copy,
                    # then uses in-round Polyak updates after each optimization step.
                    self.q_ensemble_bar = [self.build_network_from_copy(net) for net in self.q_ensemble]
                else:
                    if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                        if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                            bar_round = max(
                                (cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
                                * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"],
                                0,
                            )
                        else:
                            bar_round = max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)
                    else:
                        bar_round = max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)
                    self.load_network_bar("round_{0}_inter_{1}".format(bar_round, self.intersection_id))
            except Exception:
                print("fail to load true REDQ ensemble, current round: {0}".format(cnt_round))
                self.q_ensemble = [self.build_network() for _ in range(self.redq_n)]
                self.q_ensemble_bar = [self.build_network_from_copy(net) for net in self.q_ensemble]

        self.q_network = self.q_ensemble[0]
        self.q_network_bar = self.q_ensemble_bar[0]
        self._refresh_cos_prob_model()
        self._init_auxiliary_ssl_models()

    def _cal_len_feature(self):
        N = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            N += self._feature_dim_from_name(feat_name)
        return N

    def _feature_slice_bounds(self):
        bounds = {}
        cursor = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            dim = self._feature_dim_from_name(feat_name)
            bounds[feat_name] = (cursor, cursor + dim)
            cursor += dim
        return bounds

    def _build_static_relation_cache(self):
        if not self.neighbor_select_enabled:
            return None
        if not (self.use_topo_feature or self.use_delay_feature):
            return None
        roadnet_file = self.dic_traffic_env_conf.get("ROADNET_FILE")
        work_dir = self.dic_path.get("PATH_TO_WORK_DIRECTORY")
        if not roadnet_file or not work_dir:
            return None
        roadnet_path = os.path.join(work_dir, roadnet_file)
        if not os.path.exists(roadnet_path):
            return None
        try:
            with open(roadnet_path, "r") as f:
                net = json.load(f)
        except Exception:
            return None

        non_virtual = []
        for inter in net.get("intersections", []):
            if not inter.get("virtual", False):
                non_virtual.append(inter["id"])
        if len(non_virtual) != self.num_agents:
            return None
        inter_id_to_index = {iid: idx for idx, iid in enumerate(non_virtual)}
        coords = np.zeros((self.num_agents, 2), dtype=np.float32)
        inter_graph = {iid: set() for iid in non_virtual}
        for inter in net.get("intersections", []):
            iid = inter.get("id")
            if iid in inter_id_to_index:
                idx = inter_id_to_index[iid]
                coords[idx, 0] = float(inter["point"]["x"])
                coords[idx, 1] = float(inter["point"]["y"])
        for road in net.get("roads", []):
            start = road.get("startIntersection")
            end = road.get("endIntersection")
            if start in inter_id_to_index and end in inter_id_to_index:
                inter_graph[start].add(end)
                inter_graph[end].add(start)

        # Match cityflow_env._adjacency_extraction(): self slot + top_k-1 2-hop candidates.
        top_k = int(self.dic_traffic_env_conf.get("TOP_K_ADJACENCY", self.neighbor_topk))
        need = max(0, top_k - 1)

        def bfs_hop(src):
            dist = {src: 0}
            queue = [src]
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                cur_dist = dist[cur]
                if self.neighbor_candidate_hop >= 0 and cur_dist >= self.neighbor_candidate_hop:
                    continue
                for nxt in inter_graph.get(cur, []):
                    if nxt not in dist:
                        dist[nxt] = cur_dist + 1
                        queue.append(nxt)
            return dist

        candidate_mask_no_self = np.zeros((1, self.num_agents, self.num_agents), dtype=np.float32)
        for iid in non_virtual:
            i = inter_id_to_index[iid]
            row_dist = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
            hop_dist = bfs_hop(iid)
            allowed = [
                inter_id_to_index[j]
                for j in non_virtual
                if j != iid and hop_dist.get(j, self.neighbor_candidate_hop + 1) <= self.neighbor_candidate_hop
            ]
            allowed = sorted(allowed, key=lambda idx: row_dist[idx])[:need]
            for j in allowed:
                candidate_mask_no_self[0, i, j] = 1.0

        dx = np.abs(coords[:, None, 0] - coords[None, :, 0])
        dy = np.abs(coords[:, None, 1] - coords[None, :, 1])
        unique_dx = np.unique(dx[dx > 1e-6])
        unique_dy = np.unique(dy[dy > 1e-6])
        x_step = float(unique_dx.min()) if unique_dx.size > 0 else 1.0
        y_step = float(unique_dy.min()) if unique_dy.size > 0 else 1.0
        hop = np.round(dx / max(x_step, 1e-6)) + np.round(dy / max(y_step, 1e-6))
        adjacent = (np.abs(hop - 1.0) < 0.25).astype(np.float32)
        same_corridor = np.logical_or(dx < 1e-6, dy < 1e-6).astype(np.float32)
        distance = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-8)).astype(np.float32)

        return {
            "candidate_mask_no_self": candidate_mask_no_self.astype(np.float32),
            "hop": hop[np.newaxis, :, :, np.newaxis].astype(np.float32),
            "adjacent": adjacent[np.newaxis, :, :, np.newaxis].astype(np.float32),
            "same_corridor": same_corridor[np.newaxis, :, :, np.newaxis].astype(np.float32),
            "distance": distance[np.newaxis, :, :, np.newaxis].astype(np.float32),
        }

    def _build_static_delay_msg_cache(self):
        if not (self.use_delay_msg_mean or self.use_delay_rel_msg_mean):
            return None
        if not bool(self.dic_traffic_env_conf.get("STATIC_DELAY_CANDIDATE_MODE", False)):
            return None
        roadnet_file = self.dic_traffic_env_conf.get("ROADNET_FILE")
        work_dir = self.dic_path.get("PATH_TO_WORK_DIRECTORY")
        if not roadnet_file or not work_dir:
            return None
        roadnet_path = os.path.join(work_dir, roadnet_file)
        if not os.path.exists(roadnet_path):
            return None
        try:
            with open(roadnet_path, "r") as f:
                net = json.load(f)
        except Exception:
            return None

        non_virtual = []
        coords = {}
        for inter in net.get("intersections", []):
            if not inter.get("virtual", False):
                non_virtual.append(inter["id"])
                coords[inter["id"]] = (
                    float(inter["point"]["x"]),
                    float(inter["point"]["y"]),
                )
        if len(non_virtual) != self.num_agents:
            return None

        weighted_graph = {}
        max_lane_speed = 0.0

        def _polyline_length(points):
            if not points or len(points) < 2:
                return 0.0
            total = 0.0
            for idx in range(len(points) - 1):
                p1 = points[idx]
                p2 = points[idx + 1]
                dx = float(p1["x"]) - float(p2["x"])
                dy = float(p1["y"]) - float(p2["y"])
                total += float(np.sqrt(dx * dx + dy * dy))
            return total

        def _euclid_len(a, b):
            dx = float(a[0]) - float(b[0])
            dy = float(a[1]) - float(b[1])
            return float(np.sqrt(dx * dx + dy * dy))

        for road in net.get("roads", []):
            for lane in road.get("lanes", []):
                max_lane_speed = max(max_lane_speed, float(lane.get("maxSpeed", 0.0)))
            start = road.get("startIntersection")
            end = road.get("endIntersection")
            if start not in coords or end not in coords:
                continue
            road_len = _polyline_length(road.get("points", []))
            if road_len <= 0.0:
                road_len = _euclid_len(coords[start], coords[end])
            weighted_graph.setdefault(start, {})[end] = min(
                float(road_len),
                weighted_graph.get(start, {}).get(end, float("inf")),
            )
            weighted_graph.setdefault(end, {})[start] = min(
                float(road_len),
                weighted_graph.get(end, {}).get(start, float("inf")),
            )

        def shortest_path_lengths(src):
            dist = {src: 0.0}
            heap = [(0.0, src)]
            while heap:
                cur_dist, cur = heapq.heappop(heap)
                if cur_dist > dist.get(cur, float("inf")):
                    continue
                for nxt, weight in weighted_graph.get(cur, {}).items():
                    nd = cur_dist + float(weight)
                    if nd < dist.get(nxt, float("inf")):
                        dist[nxt] = nd
                        heapq.heappush(heap, (nd, nxt))
            return dist

        index_map = {iid: idx for idx, iid in enumerate(non_virtual)}
        min_action_time = float(self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 15))
        max_vehicle_speed = float(max_lane_speed) if max_lane_speed > 0.0 else float(
            self.dic_traffic_env_conf.get("MAX_VEHICLE_SPEED", 11.11)
        )
        tau = np.zeros((1, self.num_agents, self.num_agents, 1), dtype=np.float32)
        for src in non_virtual:
            src_idx = index_map[src]
            shortest = shortest_path_lengths(src)
            for dst, dist in shortest.items():
                dst_idx = index_map[dst]
                travel_time = float(dist) / max(max_vehicle_speed, 1e-6)
                if self.delay_msg_tau_norm_mode == "min_action_time":
                    tau_norm = travel_time / max(min_action_time, 1e-6)
                else:
                    tau_norm = travel_time
                tau[0, src_idx, dst_idx, 0] = float(tau_norm)
        return {
            "tau_norm": tau.astype(np.float32),
            "max_vehicle_speed": float(max_vehicle_speed),
            "min_action_time": float(min_action_time),
        }

    @staticmethod
    def _feature_dim_from_name(feat_name):
        if "cur_phase" in feat_name:
            return 8
        if feat_name == "intersection_topology_vector":
            return 8
        if feat_name in ("phase_elapsed", "time_this_phase", "downstream_congestion"):
            return 1
        return 12

    @staticmethod
    def _state_feature_value(state, feat_name):
        if feat_name in state:
            return state[feat_name]
        if feat_name.endswith("_previous_step"):
            base_name = feat_name[: -len("_previous_step")]
            if base_name in state:
                return state[base_name]
            if base_name == "cur_phase" and "cur_phase_previous_step" in state:
                return state["cur_phase_previous_step"]
        if feat_name == "downstream_congestion":
            return float(np.sum(state.get("lane_num_vehicle_downstream", np.zeros(12))))
        if feat_name in ("phase_elapsed", "time_this_phase"):
            return 0.0
        raise KeyError(feat_name)

    def _build_tsa_dim_mask(self):
        """
        Build [1,1,D] augmentation mask.
        By default, do NOT perturb discrete phase encoding dimensions.
        """
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        mask = np.zeros((self.len_feature,), dtype=np.float32)
        cursor = 0
        for feat_name in used_feature:
            dim = self._feature_dim_from_name(feat_name)
            use_aug = ("cur_phase" not in feat_name)
            end = min(cursor + dim, self.len_feature)
            if use_aug and end > cursor:
                mask[cursor:end] = 1.0
            cursor += dim
            if cursor >= self.len_feature:
                break
        return mask.reshape(1, 1, -1)

    def _augment_states_tsa(self, states):
        """
        Apply TSA on replayed states before Q-target/loss computation.
        states: [B, Agents, D]
        """
        if (not self.tsa_enabled) or states is None or len(states) == 0:
            return states
        x = np.array(states, dtype=np.float32, copy=True)
        dim_mask = self.tsa_dim_mask
        if dim_mask.shape[-1] != x.shape[-1]:
            # Safety fallback for unexpected feature mismatch.
            dim_mask = np.ones((1, 1, x.shape[-1]), dtype=np.float32)

        if self.tsa_gaussian_std > 0:
            noise = np.random.normal(0.0, self.tsa_gaussian_std, size=x.shape).astype(np.float32)
            x = x + noise * dim_mask

        if self.tsa_mask_prob > 0:
            drop = (np.random.rand(*x.shape) < self.tsa_mask_prob).astype(np.float32)
            x = x * (1.0 - drop * dim_mask)

        if (self.tsa_scale_low != 1.0) or (self.tsa_scale_high != 1.0):
            scale = np.random.uniform(
                self.tsa_scale_low, self.tsa_scale_high, size=(x.shape[0], x.shape[1], 1)
            ).astype(np.float32)
            x = x + (x * (scale - 1.0)) * dim_mask
        return x

    def _q_consistency_sample_weight(self, batch_n):
        if self.q_consistency_weight <= 0:
            return None
        return np.full((batch_n, self.num_agents), self.q_consistency_weight, dtype=np.float32)

    def _maybe_train_q_consistency_aux(self, net, xs_ref, xs_aug):
        if (not self.use_q_consistency_aux) or xs_ref is None or xs_aug is None:
            return
        if xs_ref[0] is None or len(xs_ref[0]) == 0:
            return
        bs = self.dic_agent_conf.get("BATCH_SIZE", 32)
        pred = net.predict(xs_ref, batch_size=bs, verbose=0)
        q_ref = np.array(pred[0] if isinstance(pred, (list, tuple)) else pred, dtype=np.float32)
        sw_aux = self._q_consistency_sample_weight(q_ref.shape[0])
        net.train_on_batch(xs_aug, q_ref, sample_weight=sw_aux)

    def _auxiliary_target_dim(self):
        if self.auxiliary_task == "latent_transition":
            return 0
        if self.auxiliary_task == "next_pressure":
            return 12
        if self.auxiliary_task == "reward":
            return 1
        return 0

    def _isr_target_dim(self):
        return self.len_feature if self.use_isr else 0

    def _pack_model_targets(self, y_main, y_aux=None, y_isr=None):
        outputs = [y_main]
        if y_aux is not None:
            outputs.append(y_aux)
        if y_isr is not None:
            outputs.append(y_isr)
        return outputs if len(outputs) > 1 else outputs[0]

    def _pack_model_sample_weight(self, q_weight=None, y_aux=None, y_isr=None):
        if y_aux is None and y_isr is None:
            return q_weight
        weights = [q_weight]
        if y_aux is not None:
            weights.append(None if q_weight is None else np.ones_like(q_weight, dtype=np.float32))
        if y_isr is not None:
            weights.append(None if q_weight is None else np.ones_like(q_weight, dtype=np.float32))
        if q_weight is None:
            return None
        return weights

    @staticmethod
    def _slice_batch_payload(payload, batch_slice):
        if payload is None:
            return None
        if isinstance(payload, (list, tuple)):
            sliced = [None if item is None else item[batch_slice] for item in payload]
            return type(payload)(sliced) if isinstance(payload, tuple) else sliced
        return payload[batch_slice]

    def _train_model_with_batches(
        self,
        net,
        inputs,
        targets,
        sample_weight,
        batch_size,
        epochs,
        verbose=False,
    ):
        sample_n = len(inputs[0]) if isinstance(inputs, (list, tuple)) else len(inputs)
        if sample_n == 0:
            return []
        losses = []
        for epoch_idx in range(max(1, int(epochs))):
            epoch_losses = []
            for start in range(0, sample_n, batch_size):
                stop = min(start + batch_size, sample_n)
                batch_slice = slice(start, stop)
                batch_inputs = self._slice_batch_payload(inputs, batch_slice)
                batch_targets = self._slice_batch_payload(targets, batch_slice)
                batch_weight = self._slice_batch_payload(sample_weight, batch_slice)
                
                # Convert to tensors before train_on_batch to strictly avoid Keras Dataset leak
                tf_inputs = [tf.convert_to_tensor(x, dtype=tf.float32) for x in batch_inputs] if isinstance(batch_inputs, list) else tf.convert_to_tensor(batch_inputs, dtype=tf.float32)
                tf_targets = [tf.convert_to_tensor(y, dtype=tf.float32) if y is not None else None for y in batch_targets] if isinstance(batch_targets, list) else tf.convert_to_tensor(batch_targets, dtype=tf.float32)
                tf_weight = [(tf.convert_to_tensor(w, dtype=tf.float32) if w is not None else None) for w in batch_weight] if isinstance(batch_weight, list) else (tf.convert_to_tensor(batch_weight, dtype=tf.float32) if batch_weight is not None else None)
                
                loss = net.train_on_batch(tf_inputs, tf_targets, sample_weight=tf_weight)
                if isinstance(loss, (list, tuple)):
                    epoch_losses.append(float(loss[0]))
                else:
                    epoch_losses.append(float(loss))
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            losses.append(mean_loss)
            if verbose:
                print(
                    "[train_on_batch-loop] epoch {}/{} mean_loss={:.6f}".format(
                        epoch_idx + 1,
                        max(1, int(epochs)),
                        mean_loss,
                    )
                )
        return losses

    def _load_init_checkpoint_if_available(self):
        init_model_dir = self.dic_path.get("PATH_TO_INIT_MODEL")
        if not init_model_dir:
            return False
        if not os.path.isdir(init_model_dir):
            print("[InitModel] skip missing init model dir {}".format(init_model_dir))
            return False
        init_file = "init_model_inter_{}".format(self.intersection_id)
        try:
            self.load_network(init_file, file_path=init_model_dir)
            if self.use_true_redq_ensemble:
                self.q_ensemble_bar = [self.build_network_from_copy(net) for net in self.q_ensemble]
                self.q_network_bar = self.q_ensemble_bar[0]
            else:
                self.q_network_bar = self.build_network_from_copy(self.q_network)
            self._refresh_cos_prob_model()
            print("[InitModel] loaded {}".format(os.path.join(init_model_dir, init_file)))
            return True
        except Exception:
            print("[InitModel] failed to load {}\n{}".format(init_model_dir, traceback.format_exc()))
            return False

    def _build_auxiliary_target(self, next_state2, reward_arr):
        if not self.use_auxiliary_head:
            return None
        if self.auxiliary_task == "next_pressure":
            bounds = self.feature_slices.get("traffic_movement_pressure_queue_efficient")
            if bounds is None:
                return None
            start, end = bounds
            return np.array(next_state2[:, :, start:end], dtype=np.float32)
        if self.auxiliary_task == "reward":
            return np.expand_dims(np.array(reward_arr, dtype=np.float32), axis=-1)
        return None

    @staticmethod
    def _q_output_only(pred):
        if isinstance(pred, (list, tuple)):
            return pred[0]
        return pred

    def _extract_queue_metric_from_sample(self, sample, reward_scalar):
        """
        Extract a queue-length-like traffic pressure metric for Triple-PER.

        Preferred order:
        1. Explicit queue metric stored in the sample tail.
        2. Raw waiting-lane counts in the serialized state.
        3. Reward inversion when the reward is pure queue_length.

        This keeps the new replay buffer compatible with existing sample files
        while still allowing future experiments to write queue_length directly.
        """
        if len(sample) > 7:
            extra = sample[7]
            if np.isscalar(extra):
                return max(float(extra), 0.0)
            if isinstance(extra, dict):
                if "queue_length" in extra:
                    return max(float(extra["queue_length"]), 0.0)
                if "lane_num_waiting_vehicle_in" in extra:
                    return max(float(np.sum(np.asarray(extra["lane_num_waiting_vehicle_in"], dtype=np.float32))), 0.0)

        state = sample[0]
        if isinstance(state, dict):
            waiting = state.get("lane_num_waiting_vehicle_in")
            if waiting is not None:
                return max(float(np.sum(np.asarray(waiting, dtype=np.float32))), 0.0)

        reward_info = self.dic_traffic_env_conf.get("DIC_REWARD_INFO", {})
        nonzero = {
            k: float(v)
            for k, v in reward_info.items()
            if v is not None and abs(float(v)) > 1e-8
        }
        if len(nonzero) == 1 and "queue_length" in nonzero:
            weight = float(nonzero["queue_length"])
            if abs(weight) > 1e-8:
                return max(float(abs(reward_scalar / weight)), 0.0)
        return 0.0

    def _compute_true_redq_triple_metrics(
        self,
        state_batch,
        adj_batch,
        next_state_batch,
        action_arr,
        reward_arr,
    ):
        """
        Compute sample-level Triple-PER metrics under the current true-REDQ ensemble.

        TD_error_mix:
            |Q_mix(s,a) - y|
            where Q_mix = (1-lambda) * mean(Q_heads) + lambda * min(Q_heads)

        Ensemble_Std:
            standard deviation of ensemble Q-values for the selected action.

        For multi-intersection batches, we aggregate each metric by mean over
        intersections so one decision step still maps to one replay priority.
        """
        bs = self.dic_agent_conf.get("BATCH_SIZE", 32)
        state_batch = np.array(state_batch, dtype=np.float32)
        next_state_batch = np.array(next_state_batch, dtype=np.float32)
        adj_batch = np.array(adj_batch, dtype=np.float32)
        action_arr = np.array(action_arr, dtype=np.int32)
        reward_arr = np.array(reward_arr, dtype=np.float32)

        q_now_list = [
            np.array(self._q_output_only(net.predict([state_batch, adj_batch], batch_size=bs, verbose=0)), dtype=np.float32)
            for net in self.q_ensemble
        ]
        q_next_target_list = [
            np.array(self._q_output_only(net.predict([next_state_batch, adj_batch], batch_size=bs, verbose=0)), dtype=np.float32)
            for net in self.q_ensemble_bar
        ]

        if self.use_double_dqn:
            q_next_online_list = [
                np.array(self._q_output_only(net.predict([next_state_batch, adj_batch], batch_size=bs, verbose=0)), dtype=np.float32)
                for net in self.q_ensemble
            ]
        else:
            q_next_online_list = None

        batch_n = int(state_batch.shape[0])
        gamma_n = float(self.dic_agent_conf["GAMMA"] ** self.nstep)
        lam = float(self.redq_lambda)

        td_err_agent = np.zeros((batch_n, self.num_agents), dtype=np.float32)
        std_agent = np.zeros((batch_n, self.num_agents), dtype=np.float32)

        for i in range(batch_n):
            for j in range(self.num_agents):
                action = int(action_arr[i, j])
                q_heads_action = np.array([q_now_list[k][i, j, action] for k in range(self.redq_n)], dtype=np.float32)
                q_mean_now = float(np.mean(q_heads_action))
                q_min_now = float(np.min(q_heads_action))
                q_mix_now = (1.0 - lam) * q_mean_now + lam * q_min_now
                std_agent[i, j] = float(np.std(q_heads_action))

                if self.use_double_dqn:
                    q_next_online = np.stack([q_next_online_list[k][i, j, :] for k in range(self.redq_n)], axis=0)
                    q_mean_online = np.mean(q_next_online, axis=0)
                    q_min_online = np.min(q_next_online, axis=0)
                    q_mix_online = (1.0 - lam) * q_mean_online + lam * q_min_online
                    best_a = int(np.argmax(q_mix_online))

                    q_target_action = np.array([q_next_target_list[k][i, j, best_a] for k in range(self.redq_n)], dtype=np.float32)
                    q_mean_tgt = float(np.mean(q_target_action))
                    q_min_tgt = float(np.min(q_target_action))
                    v_next = (1.0 - lam) * q_mean_tgt + lam * q_min_tgt
                else:
                    q_next_tgt = np.stack([q_next_target_list[k][i, j, :] for k in range(self.redq_n)], axis=0)
                    q_mean_tgt = np.mean(q_next_tgt, axis=0)
                    q_min_tgt = np.min(q_next_tgt, axis=0)
                    q_mix_tgt = (1.0 - lam) * q_mean_tgt + lam * q_min_tgt
                    v_next = float(np.max(q_mix_tgt))

                y = reward_arr[i, j] / float(self.dic_agent_conf["NORMAL_FACTOR"]) + gamma_n * v_next
                td_err_agent[i, j] = abs(float(y - q_mix_now))

        return np.mean(td_err_agent, axis=1), np.mean(std_agent, axis=1)

    def _predict_batches(self, net, inputs, batch_size):
        sample_n = len(inputs[0]) if isinstance(inputs, (list, tuple)) else len(inputs)
        preds = []
        for start in range(0, sample_n, batch_size):
            stop = min(start + batch_size, sample_n)
            batch_inputs = [tf.convert_to_tensor(x[start:stop], dtype=tf.float32) for x in inputs] if isinstance(inputs, list) else tf.convert_to_tensor(inputs[start:stop], dtype=tf.float32)
            p = net(batch_inputs, training=False)
            preds.append(self._q_output_only(p))
        return tf.concat(preds, axis=0).numpy()

    def _build_true_redq_targets_batch(
        self,
        state_batch,
        adj_batch,
        next_state_batch,
        action_arr,
        reward_arr,
    ):
        """
        Recompute true-REDQ TD targets with the *current* online/target ensembles.
        This must be called inside the UTD loop so that every optimization step uses
        a fresh target generated from the latest target networks.
        """
        if not self.use_true_redq_ensemble:
            raise RuntimeError("_build_true_redq_targets_batch requires true REDQ ensemble mode")

        bs = self.dic_agent_conf.get("BATCH_SIZE", 32)
        state_batch = np.array(state_batch, dtype=np.float32)
        next_state_batch = np.array(next_state_batch, dtype=np.float32)
        adj_batch = np.array(adj_batch, dtype=np.float32)
        action_arr = np.array(action_arr, dtype=np.int32)
        reward_arr = np.array(reward_arr, dtype=np.float32)

        target_list = [
            self._predict_batches(net, [state_batch, adj_batch], bs)
            for net in self.q_ensemble
        ]
        next_q_list = [
            self._predict_batches(net_bar, [next_state_batch, adj_batch], bs)
            for net_bar in self.q_ensemble_bar
        ]

        next_q_online_list = None
        if self.use_double_dqn:
            next_q_online_list = [
                self._predict_batches(net, [next_state_batch, adj_batch], bs)
                for net in self.q_ensemble
            ]

        slice_size = int(state_batch.shape[0])
        gamma_n = self.dic_agent_conf["GAMMA"] ** self.nstep
        lam = self.redq_lambda
        final_targets = [np.copy(t) for t in target_list]

        for i in range(slice_size):
            for j in range(self.num_agents):
                sampled = self._sample_redq_indices()
                if self.use_double_dqn:
                    q_all_online = np.stack([next_q_online_list[k][i, j, :] for k in range(self.redq_n)], axis=0)
                    q_mean_online = np.mean(q_all_online, axis=0)
                    q_sub_online = np.stack([next_q_online_list[k][i, j, :] for k in sampled], axis=0)
                    q_min_online = np.min(q_sub_online, axis=0)
                    q_mix_online = (1.0 - lam) * q_mean_online + lam * q_min_online
                    best_a = int(np.argmax(q_mix_online))

                    q_all_tgt = np.stack([next_q_list[k][i, j, best_a] for k in range(self.redq_n)])
                    q_mean_tgt = float(np.mean(q_all_tgt))
                    q_sub_tgt = np.stack([next_q_list[k][i, j, best_a] for k in sampled])
                    q_min_tgt = float(np.min(q_sub_tgt))
                    v_next = (1.0 - lam) * q_mean_tgt + lam * q_min_tgt
                else:
                    q_subset = np.stack([next_q_list[k][i, j, :] for k in sampled], axis=0)
                    q_min = np.min(q_subset, axis=0)
                    q_all = np.stack([next_q_list[k][i, j, :] for k in range(self.redq_n)], axis=0)
                    q_mean = np.mean(q_all, axis=0)
                    q_mix = (1.0 - lam) * q_mean + lam * q_min
                    v_next = float(np.max(q_mix))

                y = reward_arr[i, j] / self.dic_agent_conf["NORMAL_FACTOR"] + gamma_n * v_next
                action = int(action_arr[i, j])
                for k in range(self.redq_n):
                    final_targets[k][i, j, action] = y

        return final_targets

    def _build_ssl_encoder_views(self, q_net, suffix):
        latent_layer = q_net.get_layer("latent_repr")
        online_encoder = Model(inputs=q_net.inputs, outputs=latent_layer.output, name="ssl_encoder_{}".format(suffix))
        target_encoder = clone_model(online_encoder)
        target_encoder.set_weights(online_encoder.get_weights())
        return online_encoder, target_encoder

    def _build_ssl_transition_model(self, online_encoder, suffix):
        latent_dim = int(online_encoder.output_shape[-1])
        feature_in = Input(shape=(self.num_agents, self.len_feature), name="ssl_feature_{}".format(suffix))
        adj_in = Input(
            shape=(self.num_agents, self.num_neighbors, self.num_agents),
            name="ssl_adj_{}".format(suffix)
        )
        action_in = Input(shape=(self.num_agents, self.num_actions), name="ssl_action_{}".format(suffix))
        z_t = online_encoder([feature_in, adj_in])
        ssl_h = Concatenate(axis=-1, name="ssl_concat_{}".format(suffix))([z_t, action_in])
        ssl_h = Dense(latent_dim, activation="relu", kernel_initializer="random_normal",
                      name="ssl_transition_hidden_{}".format(suffix))(ssl_h)
        z_next = Dense(latent_dim, kernel_initializer="random_normal",
                       name="ssl_transition_out_{}".format(suffix))(ssl_h)
        model = Model(inputs=[feature_in, adj_in, action_in], outputs=z_next, name="ssl_transition_{}".format(suffix))
        model.compile(
            optimizer=Adam(lr=self.dic_agent_conf.get("LEARNING_RATE", 0.0005)),
            loss="mse",
        )
        self.ssl_latent_dim = latent_dim
        return model

    def _init_auxiliary_ssl_models(self):
        self.ssl_online_encoders = []
        self.ssl_target_encoders = []
        self.ssl_transition_models = []
        if not self.use_latent_transition_ssl:
            return
        if self.use_true_redq_ensemble:
            for q_idx, net in enumerate(self.q_ensemble):
                online_encoder, target_encoder = self._build_ssl_encoder_views(net, "q{}".format(q_idx))
                transition_model = self._build_ssl_transition_model(online_encoder, "q{}".format(q_idx))
                self.ssl_online_encoders.append(online_encoder)
                self.ssl_target_encoders.append(target_encoder)
                self.ssl_transition_models.append(transition_model)
        else:
            online_encoder, target_encoder = self._build_ssl_encoder_views(self.q_network, "single")
            transition_model = self._build_ssl_transition_model(online_encoder, "single")
            self.ssl_online_encoders.append(online_encoder)
            self.ssl_target_encoders.append(target_encoder)
            self.ssl_transition_models.append(transition_model)

    def _build_ssl_action_onehot(self, action_arr):
        eye = np.eye(self.num_actions, dtype=np.float32)
        return eye[np.array(action_arr, dtype=np.int32)]

    def _soft_update_ssl_targets(self):
        if not self.use_latent_transition_ssl:
            return
        tau = float(np.clip(self.auxiliary_ema_tau, 0.0, 1.0))
        if tau <= 0.0:
            return
        for online_encoder, target_encoder in zip(self.ssl_online_encoders, self.ssl_target_encoders):
            online_weights = online_encoder.get_weights()
            target_weights = target_encoder.get_weights()
            blended = [
                tau * target_w + (1.0 - tau) * online_w
                for online_w, target_w in zip(online_weights, target_weights)
            ]
            target_encoder.set_weights(blended)

    def _train_latent_transition_ssl(self, ssl_idx, state_inputs, next_inputs, action_onehot, sample_weight=None):
        if not self.use_latent_transition_ssl:
            return
        if state_inputs is None or next_inputs is None or action_onehot is None:
            return
        if len(state_inputs[0]) == 0:
            return
        target_encoder = self.ssl_target_encoders[ssl_idx]
        transition_model = self.ssl_transition_models[ssl_idx]
        bs = self.dic_agent_conf.get("BATCH_SIZE", 32)
        z_next_target = np.array(target_encoder.predict(next_inputs, batch_size=bs, verbose=0), dtype=np.float32)
        transition_model.train_on_batch(state_inputs + [action_onehot], z_next_target, sample_weight=sample_weight)
        self._soft_update_ssl_targets()

    def MLP(self, ins, layers=None):
        """
        MLP backbone with optional NoisyNet dense layers.
        -input: [batch,#agents,dim]
        -output: [batch,#agents,dim]
        """
        if layers is None:
            layers = [self.critic_hidden_dim] * self.critic_num_layers
        h = ins
        for layer_index, layer_size in enumerate(layers):
            dense_layer = self._dense_or_noisy(
                layer_size,
                activation=None,
                kernel_initializer="random_normal",
                name="Dense_embed_%d" % layer_index,
            )
            h = dense_layer(h)
            if self.critic_use_layer_norm:
                h = LayerNormalization(
                    epsilon=1e-6,
                    name="Dense_embed_ln_%d" % layer_index,
                )(h)
            if self.critic_activation != "linear":
                h = Activation(self.critic_activation, name="Dense_embed_act_%d" % layer_index)(h)
            if self.crossq_safe_mode and self.crossq_use_batch_norm:
                norm = self._crossq_norm_layer("Dense_embed_bn_%d" % layer_index)
                h = norm(h)
            if self.critic_dropout_rate > 0:
                h = Dropout(
                    self.critic_dropout_rate,
                    name="Dense_embed_drop_%d" % layer_index,
                )(h)
        return h

    def _feature_group_names(self):
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        groups = {"phase": [], "congestion": [], "trend": []}
        for feat_name in used_feature:
            if "cur_phase" in feat_name or feat_name in ("phase_elapsed", "time_this_phase"):
                groups["phase"].append(feat_name)
            elif (
                "pressure" in feat_name
                or "queue" in feat_name
                or "waiting" in feat_name
                or feat_name in ("lane_num_vehicle", "lane_num_vehicle_close")
            ):
                groups["congestion"].append(feat_name)
            elif (
                "running" in feat_name
                or "enter" in feat_name
                or "leave" in feat_name
                or "delta" in feat_name
                or feat_name == "downstream_congestion"
            ):
                groups["trend"].append(feat_name)
            else:
                groups["trend"].append(feat_name)
        return groups

    def _feature_group_tensor(self, feature, feat_names, group_name):
        if not feat_names:
            return Lambda(
                lambda x: tf.zeros_like(x[:, :, :1]),
                name="fg_zero_{}".format(group_name),
            )(feature)
        tensors = [
            self._slice_feature_tensor(
                feature,
                feat_name,
                fallback_dim=self._feature_dim_from_name(feat_name),
                name="fg_slice_{}_{}".format(group_name, idx),
            )
            for idx, feat_name in enumerate(feat_names)
        ]
        if len(tensors) == 1:
            return tensors[0]
        return Concatenate(name="fg_concat_raw_{}".format(group_name))(tensors)

    def _build_feature_group_encoder(self, raw_local_feature, out_dim):
        groups = self._feature_group_names()
        encoded = []
        for group_name in ("phase", "congestion", "trend"):
            group_tensor = self._feature_group_tensor(raw_local_feature, groups[group_name], group_name)
            h = Dense(
                self.feature_group_hidden_dim,
                activation="relu",
                kernel_initializer="random_normal",
                name="fg_hidden_{}".format(group_name),
            )(group_tensor)
            h = Dense(
                out_dim,
                activation="relu",
                kernel_initializer="random_normal",
                name="fg_proj_{}".format(group_name),
            )(h)
            encoded.append(h)

        if self.use_feature_group_concat:
            merged = Concatenate(name="fg_concat_encoded")(encoded)
            return Dense(
                out_dim,
                activation="relu",
                kernel_initializer="random_normal",
                name="fg_concat_out",
            )(merged)

        gate_input = Concatenate(name="fg_gate_input")(encoded)
        gate_logits = Dense(
            3,
            kernel_initializer="random_normal",
            name="fg_gate_logits",
        )(gate_input)
        gate = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name="fg_gate_softmax")(gate_logits)
        g_phase = Lambda(lambda x: x[:, :, 0:1], name="fg_gate_phase")(gate)
        g_cong = Lambda(lambda x: x[:, :, 1:2], name="fg_gate_congestion")(gate)
        g_trend = Lambda(lambda x: x[:, :, 2:3], name="fg_gate_trend")(gate)
        return Add(name="fg_gate_fused")([
            Multiply(name="fg_weighted_phase")([encoded[0], g_phase]),
            Multiply(name="fg_weighted_congestion")([encoded[1], g_cong]),
            Multiply(name="fg_weighted_trend")([encoded[2], g_trend]),
        ])

    def _build_intersection_positional_encoding(self, topo_vec, out_dim):
        """
        Full position encoder:
        - explicit coordinate harmonics
        - city-center / edge role score
        - degree / lane-length structural context
        - residual fusion back into the shared trunk
        """
        topo_vec = Lambda(
            lambda x: tf.cast(x, tf.float32),
            name="posenc_topology_cast",
        )(topo_vec)
        coord = Lambda(lambda x: x[:, :, 0:2], name="posenc_coord")(topo_vec)
        structural = Lambda(lambda x: x[:, :, 2:6], name="posenc_structural")(topo_vec)
        role = Lambda(lambda x: x[:, :, 6:8], name="posenc_role")(topo_vec)

        x_coord = Lambda(lambda x: x[:, :, 0:1], name="posenc_x")(coord)
        y_coord = Lambda(lambda x: x[:, :, 1:2], name="posenc_y")(coord)
        center_dist = Lambda(
            lambda x: tf.sqrt(tf.maximum(1e-8, tf.square(x[:, :, 0:1] - 0.5) + tf.square(x[:, :, 1:2] - 0.5))),
            name="posenc_center_dist",
        )(coord)
        center_role = Lambda(
            lambda x: tf.clip_by_value(1.0 - x / 0.70710677, 0.0, 1.0),
            name="posenc_center_role",
        )(center_dist)
        edge_role = Lambda(
            lambda x: 1.0 - x,
            name="posenc_edge_role",
        )(center_role)
        coord_harm = Concatenate(name="posenc_coord_harm")([
            x_coord,
            y_coord,
            center_role,
            edge_role,
            Lambda(lambda x: tf.sin(np.pi * x), name="posenc_sin_x")(x_coord),
            Lambda(lambda x: tf.cos(np.pi * x), name="posenc_cos_x")(x_coord),
            Lambda(lambda x: tf.sin(np.pi * x), name="posenc_sin_y")(y_coord),
            Lambda(lambda x: tf.cos(np.pi * x), name="posenc_cos_y")(y_coord),
        ])

        coord_h = Dense(
            self.intersection_pos_dim,
            activation="relu",
            kernel_initializer="random_normal",
            name="intersection_pos_coord_h",
        )(coord_harm)
        structural_h = Dense(
            self.intersection_pos_dim,
            activation="relu",
            kernel_initializer="random_normal",
            name="intersection_pos_struct_h",
        )(structural)
        role_h = Dense(
            self.intersection_pos_dim,
            activation="relu",
            kernel_initializer="random_normal",
            name="intersection_pos_role_h",
        )(role)
        fused = Concatenate(name="intersection_pos_fused")([coord_h, structural_h, role_h])
        fused = Dense(
            self.intersection_pos_dim,
            activation="relu",
            kernel_initializer="random_normal",
            name="intersection_pos_fused_h",
        )(fused)
        fused = Dense(
            out_dim,
            activation="relu",
            kernel_initializer="random_normal",
            name="intersection_pos_proj",
        )(fused)
        gate = Dense(
            out_dim,
            activation="sigmoid",
            kernel_initializer="random_normal",
            name="intersection_pos_gate",
        )(fused)
        gate_proj = Dense(
            out_dim,
            activation="relu",
            kernel_initializer="random_normal",
            name="intersection_pos_gate_proj",
        )(fused)
        return Add(name="intersection_pos_inner_residual")([
            fused,
            Multiply(name="intersection_pos_gated")([gate, gate_proj]),
        ])

    def _slice_feature_tensor(self, feature, feat_name, fallback_dim=None, name=None):
        bounds = self.feature_slices.get(feat_name)
        if bounds is None:
            dim = int(fallback_dim or 1)
            return Lambda(
                lambda x, d=dim: tf.zeros_like(x[:, :, :d]),
                name=name or "slice_zero_{}".format(feat_name),
            )(feature)
        start, end = bounds
        return Lambda(
            lambda x, s=start, e=end: x[:, :, s:e],
            name=name or "slice_{}".format(feat_name),
        )(feature)

    def _build_dynamic_collab_full_logits(self, feature, raw_local_feature):
        """
        Fuller dynamic collaborator selector:
        1. local need gate decides whether collaboration is necessary
        2. pairwise scorer ranks which collaborator is effective for the current state
        """
        pressure = self._slice_feature_tensor(
            raw_local_feature,
            "traffic_movement_pressure_queue_efficient",
            fallback_dim=12,
            name="collab_pressure_slice",
        )
        running = self._slice_feature_tensor(
            raw_local_feature,
            "lane_enter_running_part",
            fallback_dim=12,
            name="collab_running_slice",
        )
        qgr = self._slice_feature_tensor(
            raw_local_feature,
            "queue_growth_rate_movement",
            fallback_dim=12,
            name="collab_qgr_slice",
        )
        qdr = self._slice_feature_tensor(
            raw_local_feature,
            "queue_decay_rate_movement",
            fallback_dim=12,
            name="collab_qdr_slice",
        )
        topo = self._slice_feature_tensor(
            raw_local_feature,
            "intersection_topology_vector",
            fallback_dim=8,
            name="collab_topo_slice",
        )
        pressure_mean = Lambda(
            lambda x: tf.reduce_mean(tf.abs(x), axis=-1, keepdims=True),
            name="collab_pressure_mean",
        )(pressure)
        running_mean = Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
            name="collab_running_mean",
        )(running)
        qgr_mean = Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
            name="collab_qgr_mean",
        )(qgr)
        qdr_mean = Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
            name="collab_qdr_mean",
        )(qdr)

        need_in = Concatenate(name="collab_need_input")([
            feature,
            pressure_mean,
            running_mean,
            qgr_mean,
            qdr_mean,
        ])
        need_h = Dense(
            max(16, self.dynamic_collab_pair_dim // 2),
            activation="relu",
            kernel_initializer="random_normal",
            name="collab_need_h",
        )(need_in)
        need_gate = Dense(
            1,
            activation="sigmoid",
            kernel_initializer="random_normal",
            name="collab_need_gate",
        )(need_h)

        fi = Lambda(lambda x: tf.expand_dims(x, axis=2), name="collab_fi_expand")(feature)
        fj = Lambda(lambda x: tf.expand_dims(x, axis=1), name="collab_fj_expand")(feature)
        fi = Lambda(
            lambda x: tf.tile(x, [1, 1, self.num_agents, 1]),
            name="collab_fi_tile",
        )(fi)
        fj = Lambda(
            lambda x: tf.tile(x, [1, self.num_agents, 1, 1]),
            name="collab_fj_tile",
        )(fj)
        fdiff = Lambda(lambda xs: tf.abs(xs[0] - xs[1]), name="collab_fdiff")([fi, fj])
        fprod = Multiply(name="collab_fprod")([fi, fj])

        ti = Lambda(lambda x: tf.expand_dims(x, axis=2), name="collab_ti_expand")(topo)
        tj = Lambda(lambda x: tf.expand_dims(x, axis=1), name="collab_tj_expand")(topo)
        ti = Lambda(
            lambda x: tf.tile(x, [1, 1, self.num_agents, 1]),
            name="collab_ti_tile",
        )(ti)
        tj = Lambda(
            lambda x: tf.tile(x, [1, self.num_agents, 1, 1]),
            name="collab_tj_tile",
        )(tj)
        tdiff = Lambda(lambda xs: tf.abs(xs[0] - xs[1]), name="collab_tdiff")([ti, tj])

        pair_in = Concatenate(name="collab_pair_input")([fi, fj, fdiff, fprod, tdiff])
        pair_h = Dense(
            self.dynamic_collab_pair_dim,
            activation="relu",
            kernel_initializer="random_normal",
            name="collab_pair_h1",
        )(pair_in)
        pair_h = Dense(
            max(16, self.dynamic_collab_pair_dim // 2),
            activation="relu",
            kernel_initializer="random_normal",
            name="collab_pair_h2",
        )(pair_h)
        pair_logits = Dense(
            1,
            kernel_initializer="random_normal",
            name="collab_pair_score",
        )(pair_h)
        pair_logits = Lambda(lambda x: tf.squeeze(x, axis=-1), name="collab_pair_logits")(pair_logits)
        gated_logits = Lambda(
            lambda xs, b=self.dynamic_collab_need_bias: self._apply_need_gate_to_pair_logits(xs, b),
            name="cos_logits",
        )([pair_logits, need_gate])
        return gated_logits

    def _build_adaptive_phase_q_head(self, raw_feature, local_hidden):
        phase_bits = self._slice_feature_tensor(raw_feature, "cur_phase", fallback_dim=8, name="apl_phase_bits")
        pressure = self._slice_feature_tensor(
            raw_feature,
            "traffic_movement_pressure_queue_efficient",
            fallback_dim=12,
            name="apl_pressure",
        )
        running = self._slice_feature_tensor(
            raw_feature,
            "lane_enter_running_part",
            fallback_dim=12,
            name="apl_running",
        )
        if self.use_light_temporal_delta:
            delta = self._slice_feature_tensor(
                raw_feature,
                "delta_pressure",
                fallback_dim=12,
                name="apl_delta_pressure",
            )
        else:
            delta = Lambda(lambda x: tf.zeros_like(x), name="apl_delta_pressure_zero")(pressure)

        lane_order = self.dic_traffic_env_conf.get(
            "list_lane_order", ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"]
        )
        move_idx = {"WL": 0, "WT": 1, "EL": 3, "ET": 4, "NL": 6, "NT": 7, "SL": 9, "ST": 10}

        movement_scalar = {}
        movement_repr = {}
        for lane_slot, lane_name in enumerate(lane_order[:8]):
            idx = int(move_idx.get(lane_name, min(lane_slot, 11)))
            p = Lambda(
                lambda x, j=idx: x[:, :, j:j + 1],
                name="apl_pressure_{}".format(lane_name),
            )(pressure)
            r = Lambda(
                lambda x, j=idx: x[:, :, j:j + 1],
                name="apl_running_{}".format(lane_name),
            )(running)
            d = Lambda(
                lambda x, j=idx: x[:, :, j:j + 1],
                name="apl_delta_{}".format(lane_name),
            )(delta)
            ph = Lambda(
                lambda x, j=lane_slot: x[:, :, j:j + 1],
                name="apl_phasebit_{}".format(lane_name),
            )(phase_bits)

            mv_parts = [p, r, ph]
            if self.use_light_temporal_delta:
                mv_parts.append(d)
            mv_in = Concatenate(name="apl_mv_in_{}".format(lane_name))(mv_parts)
            mv_h = Dense(
                self.apl_move_hidden_dim,
                activation="relu",
                kernel_initializer="random_normal",
                name="apl_mv_h_{}".format(lane_name),
            )(mv_in)
            gate = Dense(
                1,
                activation="sigmoid",
                kernel_initializer="random_normal",
                name="apl_gate_{}".format(lane_name),
            )(mv_h)
            bias = Dense(
                1,
                activation="tanh",
                kernel_initializer="random_normal",
                name="apl_bias_{}".format(lane_name),
            )(mv_h)
            running_term = Dense(
                1,
                use_bias=False,
                kernel_initializer="random_normal",
                name="apl_run_term_{}".format(lane_name),
            )(r)
            base_scaled = Lambda(
                lambda xs: xs[0] * (0.5 + xs[1]),
                name="apl_scale_{}".format(lane_name),
            )([p, gate])
            adaptive = Add(name="apl_adapt_base_{}".format(lane_name))([base_scaled, bias, running_term])
            if self.use_light_temporal_delta:
                delta_term = Lambda(
                    lambda x, w=self.apl_temporal_delta_weight: w * x,
                    name="apl_delta_term_{}".format(lane_name),
                )(d)
                adaptive = Add(name="apl_adapt_delta_{}".format(lane_name))([adaptive, delta_term])

            movement_scalar[lane_name] = adaptive
            movement_repr[lane_name] = mv_h

        phase_list = self.dic_traffic_env_conf.get("PHASE_LIST", ["WT_ET", "NT_ST", "WL_EL", "NL_SL"])
        phase_repr = []
        for phase_name in phase_list:
            if "_" not in phase_name:
                continue
            m1, m2 = phase_name.split("_", 1)
            if m1 not in movement_scalar or m2 not in movement_scalar:
                continue
            phase_pressure = Add(name="apl_phase_pressure_{}".format(phase_name))(
                [movement_scalar[m1], movement_scalar[m2]]
            )
            phase_cat = Concatenate(name="apl_phase_cat_{}".format(phase_name))(
                [movement_repr[m1], movement_repr[m2], phase_pressure, local_hidden]
            )
            phase_h = Dense(
                self.apl_phase_hidden_dim,
                activation="relu",
                kernel_initializer="random_normal",
                name="apl_phase_h_{}".format(phase_name),
            )(phase_cat)
            phase_repr.append(phase_h)

        if len(phase_repr) != self.num_actions:
            return Dense(
                self.num_actions,
                kernel_initializer="random_normal",
                name="apl_fallback_action",
            )(local_hidden)

        phase_stack = Lambda(lambda xs: tf.stack(xs, axis=2), name="apl_phase_stack")(phase_repr)
        if self.use_light_phase_relation:
            q = Dense(
                self.apl_rel_dim,
                use_bias=False,
                kernel_initializer="random_normal",
                name="apl_rel_q",
            )(phase_stack)
            k = Dense(
                self.apl_rel_dim,
                use_bias=False,
                kernel_initializer="random_normal",
                name="apl_rel_k",
            )(phase_stack)
            v = Dense(
                self.apl_rel_dim,
                use_bias=False,
                kernel_initializer="random_normal",
                name="apl_rel_v",
            )(phase_stack)
            att_logits = Lambda(
                lambda xs, scale=np.sqrt(max(float(self.apl_rel_dim), 1.0)):
                    tf.matmul(xs[0], xs[1], transpose_b=True) / scale,
                name="apl_rel_logits",
            )([q, k])
            att = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name="apl_rel_att")(att_logits)
            ctx = Lambda(lambda xs: tf.matmul(xs[0], xs[1]), name="apl_rel_ctx")([att, v])
            phase_stack = Add(name="apl_rel_residual")([phase_stack, ctx])
            phase_stack = Dense(
                self.apl_phase_hidden_dim,
                activation="relu",
                kernel_initializer="random_normal",
                name="apl_rel_proj",
            )(phase_stack)

        if self.use_dueling:
            adv = Dense(
                1,
                kernel_initializer="random_normal",
                name="apl_advantage_scalar",
            )(phase_stack)
            adv = Lambda(lambda x: tf.squeeze(x, axis=-1), name="apl_advantage")(adv)
            pooled_phase = Lambda(lambda x: tf.reduce_mean(x, axis=2), name="apl_phase_pool")(phase_stack)
            state_cat = Concatenate(name="apl_state_cat")([pooled_phase, local_hidden])
            state_h = Dense(
                self.apl_phase_hidden_dim,
                activation="relu",
                kernel_initializer="random_normal",
                name="apl_state_h",
            )(state_cat)
            v_stream = Dense(
                1,
                kernel_initializer="random_normal",
                name="apl_value",
            )(state_h)
            return Lambda(
                lambda xs: xs[0] + xs[1] - tf.reduce_mean(xs[1], axis=-1, keepdims=True),
                name="apl_dueling_combine",
            )([v_stream, adv])

        phase_scores = Dense(
            1,
            kernel_initializer="random_normal",
            name="apl_phase_score",
        )(phase_stack)
        return Lambda(lambda x: tf.squeeze(x, axis=-1), name="apl_phase_score_sq")(phase_scores)

    @staticmethod
    def _adjacency_to_attn_mask(adj):
        """
        Convert adjacency tensor [B, N, K, N] into attention mask [B, N, N].
        Always keep self visible to avoid empty attention rows.
        """
        # Collapse sampled collaborators/neighbors dimension.
        mask = tf.reduce_sum(adj, axis=2)  # [B, N, N]
        mask = tf.cast(mask > 0, tf.float32)

        # Ensure self-attention is always allowed.
        batch = tf.shape(mask)[0]
        n = tf.shape(mask)[1]
        eye = tf.tile(tf.expand_dims(tf.eye(n, dtype=tf.float32), axis=0), [batch, 1, 1])
        mask = tf.maximum(mask, eye)
        return tf.cast(mask > 0, tf.bool)

    @staticmethod
    def _full_attn_mask_from_feature(feature):
        """
        Build full-visible attention mask [B, N, N] when no adjacency masking is desired.
        """
        batch = tf.shape(feature)[0]
        n = tf.shape(feature)[1]
        return tf.ones((batch, n, n), dtype=tf.bool)

    @staticmethod
    def _adjacency_to_row_norm_weights(adj):
        """
        Convert adjacency tensor [B, N, K, N] into row-normalized aggregation weights [B, N, N].
        """
        weights = tf.reduce_sum(adj, axis=2)  # [B, N, N]
        weights = tf.maximum(weights, 0.0)
        batch = tf.shape(weights)[0]
        n = tf.shape(weights)[1]
        eye = tf.tile(tf.expand_dims(tf.eye(n, dtype=tf.float32), axis=0), [batch, 1, 1])
        weights = tf.maximum(weights, eye)
        denom = tf.reduce_sum(weights, axis=-1, keepdims=True)
        return weights / (denom + 1e-6)

    @staticmethod
    def _adjacency_to_external_mask(adj):
        """
        Convert [B, N, K, N] slot one-hot adjacency to a boolean external-neighbor mask [B, N, N].
        Duplicated slots and self/padding are collapsed away.
        """
        mask = tf.reduce_sum(adj, axis=2)
        mask = tf.cast(mask > 0, tf.float32)
        batch = tf.shape(mask)[0]
        n = tf.shape(mask)[1]
        eye = tf.tile(tf.expand_dims(tf.eye(n, dtype=tf.float32), axis=0), [batch, 1, 1])
        return tf.maximum(mask - eye, 0.0)

    @staticmethod
    def _broadcast_source_nodes(x):
        batch = tf.shape(x)[0]
        n = tf.shape(x)[1]
        d = tf.shape(x)[2]
        return tf.broadcast_to(tf.expand_dims(x, axis=1), [batch, n, n, d])

    @staticmethod
    def _pairwise_mean_delta(x):
        x_i = tf.expand_dims(x, axis=2)
        x_j = tf.expand_dims(x, axis=1)
        return tf.reduce_mean(x_j - x_i, axis=-1, keepdims=True)

    def _tile_static_pair_tensor(self, ref_tensor, cache_np, name):
        cache_const = tf.constant(cache_np, dtype=tf.float32)
        return Lambda(
            lambda x, t=cache_const: tf.tile(t, [tf.shape(x)[0], 1, 1, 1]),
            name=name,
        )(ref_tensor)

    def _build_delay_message_mean_agg(self, h, raw_local_feature, candidate_adj, out_dim, use_rel=False, name_prefix="delay_msg"):
        pair_mask = Lambda(
            self._adjacency_to_external_mask,
            name="{}_pair_mask".format(name_prefix),
        )(candidate_adj)
        pair_mask_4d = Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            name="{}_pair_mask_4d".format(name_prefix),
        )(pair_mask)
        h_j = Lambda(
            self._broadcast_source_nodes,
            name="{}_neighbor_h".format(name_prefix),
        )(h)
        tau_pair = self._tile_static_pair_tensor(
            h,
            self.static_delay_msg_cache["tau_norm"],
            name="{}_tau_pair".format(name_prefix),
        )
        msg_parts = [h_j, tau_pair]
        if use_rel:
            pressure = self._slice_feature_tensor(
                raw_local_feature,
                "traffic_movement_pressure_queue_efficient",
                fallback_dim=12,
                name="{}_pressure".format(name_prefix),
            )
            running = self._slice_feature_tensor(
                raw_local_feature,
                "lane_enter_running_part",
                fallback_dim=12,
                name="{}_running".format(name_prefix),
            )
            if self.delay_msg_delta_reduce != "mean":
                raise ValueError("Unsupported DELAY_MSG_DELTA_REDUCE: {}".format(self.delay_msg_delta_reduce))
            delta_p = Lambda(
                self._pairwise_mean_delta,
                name="{}_delta_p".format(name_prefix),
            )(pressure)
            delta_r = Lambda(
                self._pairwise_mean_delta,
                name="{}_delta_r".format(name_prefix),
            )(running)
            msg_parts.extend([delta_p, delta_r])
        msg_in = Concatenate(axis=-1, name="{}_input".format(name_prefix))(msg_parts)
        msg_hidden = Dense(
            self.delay_msg_hidden_dim,
            activation=None if self.critic_activation == "linear" else self.critic_activation,
            kernel_initializer="random_normal",
            name="{}_dense1".format(name_prefix),
        )(msg_in)
        msg_out = Dense(
            out_dim,
            activation=None if self.critic_activation == "linear" else self.critic_activation,
            kernel_initializer="random_normal",
            name="{}_dense2".format(name_prefix),
        )(msg_hidden)
        masked_msg = Multiply(name="{}_masked".format(name_prefix))([msg_out, pair_mask_4d])
        msg_mean = Lambda(
            lambda xs: tf.reduce_sum(xs[0], axis=2) / (tf.reduce_sum(xs[1], axis=2) + 1e-6),
            name="{}_mean".format(name_prefix),
        )([masked_msg, pair_mask_4d])
        return Concatenate(axis=-1, name="{}_concat".format(name_prefix))([h, msg_mean])

    def _build_neighbor_h_mean_concat(self, h, candidate_adj, name_prefix="neighbor_h_mean"):
        pair_mask = Lambda(
            self._adjacency_to_external_mask,
            name="{}_pair_mask".format(name_prefix),
        )(candidate_adj)
        pair_mask_4d = Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            name="{}_pair_mask_4d".format(name_prefix),
        )(pair_mask)
        h_j = Lambda(
            self._broadcast_source_nodes,
            name="{}_neighbor_h".format(name_prefix),
        )(h)
        masked_h = Multiply(name="{}_masked".format(name_prefix))([h_j, pair_mask_4d])
        h_mean = Lambda(
            lambda xs: tf.reduce_sum(xs[0], axis=2) / (tf.reduce_sum(xs[1], axis=2) + 1e-6),
            name="{}_mean".format(name_prefix),
        )([masked_h, pair_mask_4d])
        return Concatenate(axis=-1, name="{}_concat".format(name_prefix))([h, h_mean])

    @staticmethod
    def _cos_probs_to_row_norm_weights(probs):
        """
        Use continuous CoS probabilities directly as collaborator aggregation weights.
        This keeps the main training path differentiable w.r.t. cos_logits.
        probs: [B, N, N]
        """
        weights = tf.maximum(probs, 0.0)
        denom = tf.reduce_sum(weights, axis=-1, keepdims=True)
        return weights / (denom + 1e-6)

    @staticmethod
    def _cos_probs_to_topk_row_norm_weights(xs):
        """
        Combine a learned CoS probability matrix with the selected dynamic adjacency.

        xs = [probs, adj]
        probs: [B, N, N] soft collaborator probabilities
        adj:   [B, N, K, N] selected collaborator slots from CoSDynamicAdjacency

        The resulting weights preserve soft weighting within the chosen top-k set,
        while strictly masking out non-selected collaborators.
        """
        probs, adj = xs
        probs = tf.maximum(probs, 0.0)
        mask = tf.reduce_sum(adj, axis=2)  # [B, N, N]
        mask = tf.cast(mask > 0, probs.dtype)
        batch = tf.shape(probs)[0]
        n = tf.shape(probs)[1]
        eye = tf.tile(tf.expand_dims(tf.eye(n, dtype=probs.dtype), axis=0), [batch, 1, 1])
        mask = tf.maximum(mask, eye)
        weights = probs * mask
        denom = tf.reduce_sum(weights, axis=-1, keepdims=True)
        return weights / (denom + 1e-6)

    @staticmethod
    def _mask_scores_with_candidate_adj(xs):
        scores, adj = xs
        scores = tf.cast(scores, tf.float32)
        cand_mask = tf.reduce_sum(adj, axis=2)  # [B,N,N]
        cand_mask = tf.cast(cand_mask > 0, scores.dtype)
        batch = tf.shape(scores)[0]
        n = tf.shape(scores)[1]
        eye = tf.tile(tf.expand_dims(tf.eye(n, dtype=scores.dtype), axis=0), [batch, 1, 1])
        cand_mask = tf.maximum(cand_mask, eye)
        neg_large = tf.constant(-1e9, dtype=scores.dtype)
        return tf.where(cand_mask > 0, scores, neg_large)

    @staticmethod
    def _apply_need_gate_to_pair_logits(xs, self_bias):
        """
        Gate off-diagonal collaborator scores by a learned local need signal.
        When need is low, non-self collaborator logits are strongly suppressed and
        self-collaboration gets an additive bias.
        """
        logits, need = xs  # logits:[B,N,N], need:[B,N,1]
        logits = tf.cast(logits, tf.float32)
        need = tf.cast(need, logits.dtype)
        need = tf.clip_by_value(need, 1e-4, 1.0)
        batch = tf.shape(logits)[0]
        n = tf.shape(logits)[1]
        eye = tf.tile(tf.expand_dims(tf.eye(n, dtype=logits.dtype), axis=0), [batch, 1, 1])
        offdiag = 1.0 - eye
        need_row = tf.tile(need, [1, 1, n])
        gated = logits + offdiag * tf.math.log(need_row)
        gated = gated + eye * (tf.constant(self_bias, dtype=logits.dtype) * (1.0 - need_row))
        return gated

    @staticmethod
    def _adjacency_to_binary_mask(adj):
        mask = tf.reduce_sum(adj, axis=2)
        mask = tf.cast(mask > 0, tf.float32)
        batch = tf.shape(mask)[0]
        n = tf.shape(mask)[1]
        eye = tf.tile(tf.expand_dims(tf.eye(n, dtype=tf.float32), axis=0), [batch, 1, 1])
        return tf.maximum(mask, eye)

    def _phase_bits_to_action_index(self, phase_bits):
        phase_cfg = self.dic_traffic_env_conf.get("PHASE", {})
        phase_vecs = []
        for idx in range(1, self.num_actions + 1):
            vec = phase_cfg.get(idx, phase_cfg.get(str(idx), [0] * 8))
            phase_vecs.append(vec)
        phase_tensor = tf.constant(np.array(phase_vecs, dtype=np.float32))
        diff = tf.reduce_sum(
            tf.square(tf.expand_dims(phase_bits, axis=2) - tf.reshape(phase_tensor, [1, 1, self.num_actions, 8])),
            axis=-1,
        )
        return tf.argmin(diff, axis=-1)

    def _transformer_encoder_stack(self, x, attn_mask, d_model):
        """
        CoSLight-style Transformer encoder stack on top of agent features.
        x: [B, N, D], attn_mask: [B, N, N]
        """
        h = x
        history_states = [h]
        key_dim = max(1, d_model // max(1, self.trans_heads))
        for layer_idx in range(self.trans_layers):
            mha = MultiHeadAttention(
                num_heads=self.trans_heads,
                key_dim=key_dim,
                dropout=self.trans_dropout,
                name="trans_mha_{}".format(layer_idx),
            )
            drop1 = Dropout(self.trans_dropout, name="trans_drop1_{}".format(layer_idx))
            drop2 = Dropout(self.trans_dropout, name="trans_drop2_{}".format(layer_idx))
            drop3 = Dropout(self.trans_dropout, name="trans_drop3_{}".format(layer_idx))
            ff1 = Dense(
                self.trans_ffn_dim,
                activation="relu",
                kernel_initializer="random_normal",
                name="trans_ffn1_{}".format(layer_idx),
            )
            ff2 = Dense(
                d_model,
                kernel_initializer="random_normal",
                name="trans_ffn2_{}".format(layer_idx),
            )

            if self.trans_prenorm:
                x1 = LayerNormalization(epsilon=1e-6, name="trans_ln1_{}".format(layer_idx))(h)
                attn_out = mha(query=x1, value=x1, key=x1, attention_mask=attn_mask)
                attn_out = drop1(attn_out)
                h = Add(name="trans_add1_{}".format(layer_idx))([h, attn_out])

                x2 = LayerNormalization(epsilon=1e-6, name="trans_ln2_{}".format(layer_idx))(h)
                ffn_out = ff1(x2)
                ffn_out = drop2(ffn_out)
                ffn_out = ff2(ffn_out)
                ffn_out = drop3(ffn_out)
                h = Add(name="trans_add2_{}".format(layer_idx))([h, ffn_out])
            else:
                attn_out = mha(query=h, value=h, key=h, attention_mask=attn_mask)
                attn_out = drop1(attn_out)
                h = Add(name="trans_add1_{}".format(layer_idx))([h, attn_out])
                h = LayerNormalization(epsilon=1e-6, name="trans_ln1_{}".format(layer_idx))(h)

                ffn_out = ff1(h)
                ffn_out = drop2(ffn_out)
                ffn_out = ff2(ffn_out)
                ffn_out = drop3(ffn_out)
                h = Add(name="trans_add2_{}".format(layer_idx))([h, ffn_out])
                h = LayerNormalization(epsilon=1e-6, name="trans_ln2_{}".format(layer_idx))(h)
            if self.use_block_attn_res:
                hist_inputs = history_states + [h]
                hist_stack = Lambda(
                    lambda xs: tf.stack(xs, axis=2),
                    name="trans_hist_stack_{}".format(layer_idx),
                )(hist_inputs)
                query_expand = Lambda(
                    lambda t: tf.expand_dims(t, axis=2),
                    name="trans_hist_query_expand_{}".format(layer_idx),
                )(h)
                query_tiled = Lambda(
                    lambda t, hist_len=len(hist_inputs): tf.repeat(t, repeats=hist_len, axis=2),
                    name="trans_hist_query_tile_{}".format(layer_idx),
                )(query_expand)
                score_in = Concatenate(axis=-1, name="trans_hist_score_in_{}".format(layer_idx))(
                    [hist_stack, query_tiled]
                )
                scores = Dense(
                    1,
                    kernel_initializer="random_normal",
                    name="trans_hist_score_{}".format(layer_idx),
                )(score_in)
                weights = Lambda(
                    lambda t: tf.nn.softmax(t, axis=2),
                    name="trans_hist_softmax_{}".format(layer_idx),
                )(scores)
                h = Lambda(
                    lambda xs: tf.reduce_sum(xs[0] * xs[1], axis=2),
                    name="trans_hist_weighted_sum_{}".format(layer_idx),
                )([hist_stack, weights])
            history_states.append(h)
        return h

    def MultiHeadsAttModel(self, in_feats, in_nei, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
        """
        input: [batch, agent, dim] feature
               [batch, agent, nei, agent] adjacency
        input:[bacth,agent,128]
        output:
              [batch, agent, dim]
        """
        # [batch,agent,dim]->[batch,agent,1,dim]
        agent_repr = Reshape((self.num_agents, 1, d_in))(in_feats)

        # [batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr = RepeatVector3D(self.num_agents)(in_feats)

        # [batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        neighbor_repr = MatMulLayer()([in_nei, neighbor_repr])

        # attention computation
        # [batch, agent, 1, dim]->[batch, agent, 1, h_dim*head]
        agent_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                name='agent_repr_%d' % suffix)(agent_repr)
        # [batch,agent,1,h_dim,head]->[batch,agent,head,1,h_dim]
        agent_repr_head = Reshape((self.num_agents, 1, h_dim, head))(agent_repr_head)
        agent_repr_head = PermuteDimensionsLayer()(agent_repr_head)

        # [batch,agent,neighbor,dim]->[batch,agent,neighbor,h_dim_head]
        neighbor_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                   name='neighbor_repr_%d' % suffix)(neighbor_repr)
        # [batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
        neighbor_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(neighbor_repr_head)
        neighbor_repr_head = PermuteDimensionsLayer()(neighbor_repr_head)

        # [batch,agent,head,1,h_dim]x[batch,agent,head,neighbor,h_dim]->[batch,agent,head,1,neighbor]
        att = SoftmaxMatMulLayer()([agent_repr_head,
                                   neighbor_repr_head])
        # [batch,agent,nv,1,neighbor]->[batch,agent,head,neighbor]
        att_record = Reshape((self.num_agents, head, self.num_neighbors))(att)

        # self embedding again
        neighbor_hidden_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                          name='neighbor_hidden_repr_%d' % suffix)(neighbor_repr)
        neighbor_hidden_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(
            neighbor_hidden_repr_head)
        neighbor_hidden_repr_head = PermuteDimensionsLayer()(
            neighbor_hidden_repr_head)
        out = MeanMatMulLayer()([att, neighbor_hidden_repr_head])
        out = Reshape((self.num_agents, h_dim))(out)
        out = Dense(dout, activation="relu", kernel_initializer='random_normal', name='MLP_after_relation_%d' % suffix)(
            out)
        return out, att_record

    def CompetitiveAttModel(self, in_feats, in_nei, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
        """
        CityLight-inspired competitive neighbor aggregation.
        Splits neighbors into 2 competing groups, applies separate attention
        within each group, then concatenates group outputs.
        adjacency_row = [self, n1, n2, n3, n4] → group1=[self,n1,n2], group2=[self,n3,n4]
        In grid layout: first 2 neighbors ≈ EW (opposing), last 2 ≈ NS (opposing).
        """
        num_nei = self.num_neighbors  # typically 5 (self+4)
        mid = (num_nei + 1) // 2  # split: first mid neighbors vs rest
        # group sizes (including self at idx 0 in both via separate agent_repr)
        g1_size = mid   # neighbors [0..mid-1]
        g2_size = num_nei - mid  # neighbors [mid..num_nei-1]

        # --- shared feature extraction ---
        agent_repr = Reshape((self.num_agents, 1, d_in))(in_feats)
        neighbor_repr_all = RepeatVector3D(self.num_agents)(in_feats)
        neighbor_repr_all = MatMulLayer()([in_nei, neighbor_repr_all])
        # neighbor_repr_all: [batch, agents, num_nei, d_in]

        # --- split neighbors into two groups ---
        group1_repr = SliceLayer(start=0, end=mid,
                                 name='group1_slice_%d' % suffix)(neighbor_repr_all)
        group2_repr = SliceLayer(start=mid, end=None,
                                 name='group2_slice_%d' % suffix)(neighbor_repr_all)

        group_outs = []
        head_per_group = max(head // 2, 1)
        for g_idx, (g_repr, g_size) in enumerate([(group1_repr, g1_size), (group2_repr, g2_size)]):
            gname = 'g%d_l%d' % (g_idx, suffix)
            # query: agent self
            q = Dense(h_dim * head_per_group, activation='relu', kernel_initializer='random_normal',
                      name='q_%s' % gname)(agent_repr)
            q = Reshape((self.num_agents, 1, h_dim, head_per_group))(q)
            q = PermuteDimensionsLayer()(q)

            # key: group neighbors
            k = Dense(h_dim * head_per_group, activation='relu', kernel_initializer='random_normal',
                      name='k_%s' % gname)(g_repr)
            k = Reshape((self.num_agents, g_size, h_dim, head_per_group))(k)
            k = PermuteDimensionsLayer()(k)

            # attention scores
            att = SoftmaxMatMulLayer()([q, k])

            # value: group neighbors
            v = Dense(h_dim * head_per_group, activation='relu', kernel_initializer='random_normal',
                      name='v_%s' % gname)(g_repr)
            v = Reshape((self.num_agents, g_size, h_dim, head_per_group))(v)
            v = PermuteDimensionsLayer()(v)

            g_out = MeanMatMulLayer()([att, v])
            g_out = Reshape((self.num_agents, h_dim))(g_out)
            group_outs.append(g_out)

        # concatenate competing group outputs
        merged = Concatenate(axis=-1, name='compete_merge_%d' % suffix)(group_outs)
        out = Dense(dout, activation="relu", kernel_initializer='random_normal',
                    name='MLP_after_compete_%d' % suffix)(merged)
        # dummy att_record for interface compatibility
        att_record = ZerosLayer(num_agents=self.num_agents, head=head, num_neighbors=self.num_neighbors,
                                name='dummy_att_%d' % suffix)(in_feats)
        return out, att_record

    def adjacency_index2matrix(self, adjacency_index):
        # [batch, agents, neighbors]
        adjacency_index_new = np.array(adjacency_index, copy=True)
        cur_k = adjacency_index_new.shape[-1]
        target_k = int(self.num_neighbors)
        if cur_k > target_k:
            adjacency_index_new = adjacency_index_new[..., :target_k]
        elif cur_k < target_k:
            # Pad missing collaborator slots with self ids so the tensor shape always
            # matches the model input when dynamic CoS adjacency uses a smaller K.
            self_ids = adjacency_index_new[..., :1]
            pad = np.repeat(self_ids, target_k - cur_k, axis=-1)
            adjacency_index_new = np.concatenate([adjacency_index_new, pad], axis=-1)
        adjacency_index_new = np.sort(adjacency_index_new, axis=-1)
        lab = to_categorical(adjacency_index_new, num_classes=self.num_agents)
        return lab

    def convert_state_to_input(self, s):
        """
        s: [state1, state2, ..., staten]
        """
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        flat_feature = list(used_feature)
        feats0 = []
        adj = []
        for i in range(self.num_agents):
            adj.append(s[i]["adjacency_matrix"])
            tmp = []
            for feature in flat_feature:
                if "cur_phase" in feature:
                    if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                        # Resume loads traffic_env.conf from JSON, where dict keys become strings.
                        # Support both int and str phase keys to avoid KeyError during resumed runs.
                        phase_cfg = self.dic_traffic_env_conf['PHASE']
                        phase_value = self._state_feature_value(s[i], feature)
                        phase_id = phase_value[0]
                        phase_vec = phase_cfg.get(phase_id)
                        if phase_vec is None:
                            phase_vec = phase_cfg.get(str(phase_id))
                        if phase_vec is None:
                            raise KeyError("Unknown phase id {} in PHASE config".format(phase_id))
                        tmp.extend(phase_vec)
                    else:
                        value = self._state_feature_value(s[i], feature)
                        if np.isscalar(value):
                            tmp.append(float(value))
                        else:
                            tmp.extend(value)
                else:
                    value = self._state_feature_value(s[i], feature)
                    if np.isscalar(value):
                        tmp.append(float(value))
                    else:
                        tmp.extend(value)

            feats0.append(tmp)
        feats = np.array([feats0])
        adj = self.adjacency_index2matrix(np.array([adj]))
        return [feats, adj]

    def _refresh_cos_prob_model(self):
        """Build a probe model for CoS probabilities if the layer exists."""
        try:
            self.cos_prob_model = Model(
                inputs=self.q_network.inputs,
                outputs=self.q_network.get_layer("cos_probs").output
            )
        except Exception:
            self.cos_prob_model = None

    def _topk_cos_ids_from_probs(self, probs):
        """
        probs: [B, N, N]
        return ids: [B, N, K] (K includes self when enabled).
        """
        bsz = probs.shape[0]
        n = probs.shape[1]
        k = self.num_neighbors
        other_k = max(0, k - (1 if self.cos_include_self else 0))
        other_k = min(other_k, max(0, n - (1 if self.cos_include_self else 0)))

        probs_work = probs.copy()
        if self.cos_include_self:
            for i in range(n):
                probs_work[:, i, i] = -1.0

        if other_k > 0:
            topk_others = np.argpartition(-probs_work, kth=other_k - 1, axis=-1)[:, :, :other_k]
        else:
            topk_others = np.zeros((bsz, n, 0), dtype=np.int32)

        if self.cos_include_self:
            self_ids = np.tile(np.arange(n, dtype=np.int32).reshape(1, n, 1), (bsz, 1, 1))
            ids = np.concatenate([self_ids, topk_others.astype(np.int32)], axis=-1)
        else:
            ids = topk_others.astype(np.int32)
        return ids

    def choose_action(self, count, states):
        """
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        """
        xs = self.convert_state_to_input(states)
        use_epsilon = not self.use_noisy_net
        if self.use_true_redq_ensemble:
            q_values = []
            for net in self.q_ensemble:
                pred = net(xs)
                q_pred = self._q_output_only(pred)
                q_arr = np.array(q_pred, dtype=np.float32)
                if q_arr.ndim == 3:
                    q_arr = q_arr[0]
                q_values.append(q_arr)  # [Agents, A]
            q_stack = np.stack(q_values, axis=0)  # [N, Agents, A]
            if use_epsilon and random.random() <= self.dic_agent_conf["EPSILON"]:
                action = np.random.randint(self.num_actions, size=q_stack.shape[1])
                return action
            if self.relight_action_vote:
                # RELight-style acting: each critic votes an argmax action.
                q_mean = np.mean(q_stack, axis=0)  # [Agents, A], used for tie break.
                votes = np.argmax(q_stack, axis=2)  # [N, Agents]
                actions = np.zeros((q_stack.shape[1],), dtype=np.int32)
                for ag in range(q_stack.shape[1]):
                    counts = np.bincount(votes[:, ag], minlength=self.num_actions)
                    max_cnt = np.max(counts)
                    cand = np.where(counts == max_cnt)[0]
                    if cand.size == 1:
                        actions[ag] = int(cand[0])
                    else:
                        # Deterministic tie-break by mean Q on candidate actions.
                        cand_q = q_mean[ag, cand]
                        actions[ag] = int(cand[np.argmax(cand_q)])
                return actions
            q_mean = np.mean(q_stack, axis=0)  # [Agents, A]
            if self.dic_agent_conf.get("DETERMINISTIC_REDQ_ACTING", False):
                # Keep exploration in epsilon-greedy, but make the greedy policy
                # itself deterministic by aggregating over all critics.
                q_min = np.min(q_stack, axis=0)  # [Agents, A]
            else:
                sampled = self._sample_redq_indices()
                q_min = np.min(q_stack[sampled], axis=0)  # [Agents, A]
            q_policy = (1.0 - self.redq_lambda) * q_mean + self.redq_lambda * q_min
            if self.use_ucb_action:
                q_std = np.std(q_stack, axis=0)  # [Agents, A]
                ucb_coef = max(self.ucb_min, self.ucb_lambda * pow(self.ucb_decay, count))
                q_policy = q_policy + ucb_coef * q_std
            q_policy = self._apply_action_gaussian_noise(q_policy)
            action = np.argmax(q_policy, axis=1)
            return action

        q_values = self._q_output_only(self.q_network(xs))
        if self.cos_enabled and self.cos_prob_model is not None and self.head_debug and count % 50 == 0:
            probs = np.array(self.cos_prob_model.predict(xs, verbose=0))
            ids = self._topk_cos_ids_from_probs(probs)
            print("[CoS] sample ids(inter0):", ids[0, 0].tolist())
        if self.use_multihead:
            # q_values: [1, Agents, N, A]
            q_heads = np.array(q_values[0], dtype=np.float32)  # [Agents, N, A]
            q_mean = np.mean(q_heads, axis=1)  # [Agents, A]
            q_policy = q_mean
            if self.use_redq and self.true_redq_mode:
                # True REDQ action selection: use REDQ-style Q_mix, not plain head mean.
                m = max(1, min(self.redq_m, self.head_n))
                q_policy = np.zeros_like(q_mean, dtype=np.float32)
                for a in range(q_heads.shape[0]):
                    if m >= self.head_n:
                        sampled_q = q_heads[a]  # [N, A]
                    else:
                        sampled_heads = np.random.choice(self.head_n, m, replace=False)
                        sampled_q = q_heads[a, sampled_heads, :]  # [M, A]
                    q_min = np.min(sampled_q, axis=0)  # [A]
                    q_policy[a] = (1.0 - self.redq_lambda) * q_mean[a] + self.redq_lambda * q_min
            if self.use_ucb_action:
                q_std = np.std(q_heads, axis=1)  # [Agents, A]
                ucb_coef = max(self.ucb_min, self.ucb_lambda * pow(self.ucb_decay, count))
                q_policy = q_mean + ucb_coef * q_std
            if self.head_debug:
                per_head_actions = np.argmax(q_heads, axis=2)  # [Agents, N]
                agree = np.mean([len(set(per_head_actions[a])) == 1
                                 for a in range(per_head_actions.shape[0])])
                print("[MultiHead] head agreement rate: {:.2f}".format(agree))
            if use_epsilon and random.random() <= self.dic_agent_conf["EPSILON"]:
                action = np.random.randint(self.num_actions, size=len(q_mean))
            else:
                q_policy = self._apply_action_gaussian_noise(q_policy)
                action = np.argmax(q_policy, axis=1)
        else:
            if use_epsilon and random.random() <= self.dic_agent_conf["EPSILON"]:
                action = np.random.randint(self.num_actions, size=len(q_values[0]))
            else:
                q_policy = np.array(q_values[0], dtype=np.float32)
                q_policy = self._apply_action_gaussian_noise(q_policy)
                action = np.argmax(q_policy, axis=1)
        return action

    @staticmethod
    def _concat_list(ls):
        tmp = []
        for i in range(len(ls)):
            if np.isscalar(ls[i]):
                tmp.append(float(ls[i]))
            else:
                tmp += ls[i]
        return [tmp]

    def _apply_action_gaussian_noise(self, q_policy):
        std = float(getattr(self, "action_gaussian_std", 0.0))
        if std <= 0:
            return q_policy
        noise = np.random.normal(0.0, std, size=q_policy.shape).astype(np.float32)
        clip = float(getattr(self, "action_gaussian_clip", 0.0))
        if clip > 0:
            noise = np.clip(noise, -clip, clip)
        return q_policy + noise

    def _dense_or_noisy(self, units, activation=None, kernel_initializer="random_normal", name=None, use_bias=True):
        if self.use_noisy_net:
            return NoisyDense(
                units,
                activation=activation,
                sigma_init=self.noisy_sigma_init,
                use_bias=use_bias,
                name=name,
            )
        return Dense(
            units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            name=name,
        )

    def _per_resample_indices(
        self,
        state_batch,
        adj_batch,
        next_state_batch,
        action_arr,
        reward_arr,
    ):
        """
        Proportional PER with TD-error priorities on the candidate pool.
        Priority for each time-step sample is the mean TD-error across intersections.
        """
        pool_size = state_batch.shape[0]
        target_size = min(int(self.dic_agent_conf["SAMPLE_SIZE"]), pool_size)
        if (not self.use_per) or pool_size <= target_size:
            return np.arange(pool_size, dtype=np.int32)
        if getattr(self, "cnt_round", 0) < self.per_warmup_rounds:
            self.per_is_weights = None
            return np.random.choice(pool_size, size=target_size, replace=False)

        if self.use_true_redq_ensemble:
            inputs_now = [state_batch, adj_batch]
            inputs_next = [next_state_batch, adj_batch]
            q_now = np.array(self.q_ensemble[0](inputs_now), dtype=np.float32)
            q_next = np.array(self.q_ensemble_bar[0](inputs_next), dtype=np.float32)
        else:
            inputs_now = [state_batch, adj_batch]
            inputs_next = [next_state_batch, adj_batch]
            q_now = np.array(self.q_network(inputs_now), dtype=np.float32)
            q_next = np.array(self.q_network_bar(inputs_next), dtype=np.float32)

        if q_now.ndim == 4:
            q_now = np.mean(q_now, axis=2)
        if q_next.ndim == 4:
            q_next = np.mean(q_next, axis=2)

        batch_idx = np.arange(pool_size)[:, None]
        agent_idx = np.arange(self.num_agents)[None, :]
        pred_q = q_now[batch_idx, agent_idx, action_arr]
        target_q = reward_arr / float(self.dic_agent_conf["NORMAL_FACTOR"]) + \
            float(self.dic_agent_conf["GAMMA"]) * np.max(q_next, axis=2)
        td_err = np.abs(target_q - pred_q)
        sample_pri = np.mean(td_err, axis=1)

        pri_base = sample_pri + self.per_eps
        if self.per_priority_clip > 0:
            pri_base = np.minimum(pri_base, self.per_priority_clip)
        pri = np.power(pri_base, self.per_alpha)
        pri_sum = float(np.sum(pri))
        if not np.isfinite(pri_sum) or pri_sum <= 0:
            probs = np.ones(pool_size, dtype=np.float64) / float(pool_size)
        else:
            probs = pri / pri_sum
        if self.per_uniform_mix > 0:
            uni = 1.0 / float(pool_size)
            probs = (1.0 - self.per_uniform_mix) * probs + self.per_uniform_mix * uni
            probs = probs / np.sum(probs)

        # Calculate Importance Sampling (IS) weights for selected indices
        chosen_idx = np.random.choice(pool_size, size=target_size, replace=False, p=probs)
        per_beta = float(self.dic_agent_conf.get("PER_BETA", 0.4))
        if per_beta > 0 and pool_size > 1:
            weights = np.power(pool_size * probs[chosen_idx], -per_beta)
            weights /= np.max(weights) if np.max(weights) > 0 else 1.0
            self.per_is_weights = weights
        else:
            self.per_is_weights = None
        return chosen_idx

    def prepare_Xs_Y(self, memory):
        """
        memory: [slice_data, slice_data, ..., slice_data]
        prepare memory for training
        """
        if len(memory) == 0 or len(memory[0]) == 0:
            self.Xs, self.Y = [], []
            self.Y_aux = None
            self.Y_isr = None
            self.ssl_action_onehot = None
            self.ssl_next_inputs = None
            self.ssl_sample_weight = None
            self.redq_next_state_train = None
            self.redq_action_arr = None
            self.redq_reward_arr = None
            return
        slice_size = len(memory[0])
        _adjs = []
        _state = [[] for _ in range(self.num_agents)]
        _next_state = [[] for _ in range(self.num_agents)]
        _action = [[] for _ in range(self.num_agents)]
        _reward = [[] for _ in range(self.num_agents)]
        _queue = [[] for _ in range(self.num_agents)]

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        flat_feature = list(used_feature)

        for i in range(slice_size):
            _adj = []
            for j in range(self.num_agents):
                sample = memory[j][i]
                state = sample[0]
                action = sample[1]
                next_state = sample[2]
                reward = sample[3]
                _action[j].append(action)
                _reward[j].append(reward)
                _queue[j].append(self._extract_queue_metric_from_sample(sample, reward))
                _adj.append(state["adjacency_matrix"])
                _state[j].append(self._concat_list([
                    self._state_feature_value(state, flat_feature[idx]) for idx in range(len(flat_feature))
                ]))
                _next_state[j].append(self._concat_list([
                    self._state_feature_value(next_state, flat_feature[idx]) for idx in range(len(flat_feature))
                ]))
            _adjs.append(_adj)
        # [batch, agent, nei, agent]
        _adjs2 = self.adjacency_index2matrix(np.array(_adjs))

        # [batch, 1, dim] -> [batch, agent, dim]
        _state2 = np.concatenate([np.array(ss) for ss in _state], axis=1).astype(np.float32)
        _next_state2 = np.concatenate([np.array(ss) for ss in _next_state], axis=1).astype(np.float32)
        _action_arr = np.array(_action, dtype=np.int32).transpose(1, 0)  # [B, Agents]
        _reward_arr = np.array(_reward, dtype=np.float32).transpose(1, 0)  # [B, Agents]
        _queue_arr = np.array(_queue, dtype=np.float32).transpose(1, 0)  # [B, Agents]

        # Optional PER resampling from candidate pool.
        keep_idx = self._per_resample_indices(
            _state2,
            _adjs2,
            _next_state2,
            _action_arr,
            _reward_arr,
        )
        keep_idx = np.array(keep_idx, dtype=np.int32)
        if keep_idx.shape[0] != slice_size:
            _state2 = _state2[keep_idx]
            _next_state2 = _next_state2[keep_idx]
            _adjs2 = _adjs2[keep_idx]
            _action_arr = _action_arr[keep_idx]
            _reward_arr = _reward_arr[keep_idx]
            slice_size = int(keep_idx.shape[0])
            _action = [_action_arr[:, j].tolist() for j in range(self.num_agents)]
            _reward = [_reward_arr[:, j].tolist() for j in range(self.num_agents)]

        _state2_train = self._augment_states_tsa(_state2)
        if self.tsa_apply_to_next_state:
            _next_state2_train = self._augment_states_tsa(_next_state2)
        else:
            _next_state2_train = _next_state2
        _state2_aux = self._augment_states_tsa(_state2) if self.use_q_consistency_aux else None
        _aux_target = self._build_auxiliary_target(_next_state2, _reward_arr)
        _isr_target = np.array(_next_state2, dtype=np.float32) if self.use_isr else None
        _ssl_action_onehot = self._build_ssl_action_onehot(_action_arr) if self.use_latent_transition_ssl else None
        _ssl_next_inputs = ([_next_state2, _adjs2] if self.use_latent_transition_ssl else None)
        if self.use_true_redq_ensemble:
            # True REDQ: independent critic ensemble.
            bs = self.dic_agent_conf.get("BATCH_SIZE", 32)
            target_list = [
                np.array(self._q_output_only(net.predict([_state2_train, _adjs2], batch_size=bs, verbose=0)),
                         dtype=np.float32)
                for net in self.q_ensemble
            ]  # N x [B, Agents, A]
            next_q_list = [
                np.array(self._q_output_only(net_bar.predict([_next_state2_train, _adjs2], batch_size=bs, verbose=0)),
                         dtype=np.float32)
                for net_bar in self.q_ensemble_bar
            ]  # N x [B, Agents, A]

            if self.use_double_dqn:
                # Double DQN: use online ensemble to select action, target ensemble to evaluate.
                next_q_online_list = [
                    np.array(self._q_output_only(net.predict([_next_state2_train, _adjs2], batch_size=bs, verbose=0)),
                             dtype=np.float32)
                    for net in self.q_ensemble
                ]  # N x [B, Agents, A]

            gamma_n = self.dic_agent_conf["GAMMA"] ** self.nstep
            final_targets = [np.copy(t) for t in target_list]
            lam = self.redq_lambda
            for i in range(slice_size):
                for j in range(self.num_agents):
                    sampled = self._sample_redq_indices()
                    if self.use_double_dqn:
                        # Online network selects greedy action (Q_mix from online ensemble).
                        q_all_online = np.stack([next_q_online_list[k][i, j, :] for k in range(self.redq_n)], axis=0)
                        q_mean_online = np.mean(q_all_online, axis=0)
                        q_sub_online = np.stack([next_q_online_list[k][i, j, :] for k in sampled], axis=0)
                        q_min_online = np.min(q_sub_online, axis=0)
                        q_mix_online = (1.0 - lam) * q_mean_online + lam * q_min_online
                        best_a = int(np.argmax(q_mix_online))
                        # Target network evaluates that action only.
                        q_all_tgt = np.stack([next_q_list[k][i, j, best_a] for k in range(self.redq_n)])
                        q_mean_tgt = float(np.mean(q_all_tgt))
                        q_sub_tgt = np.stack([next_q_list[k][i, j, best_a] for k in sampled])
                        q_min_tgt = float(np.min(q_sub_tgt))
                        v_next = (1.0 - lam) * q_mean_tgt + lam * q_min_tgt
                    else:
                        q_subset = np.stack([next_q_list[k][i, j, :] for k in sampled], axis=0)  # [M, A]
                        q_min = np.min(q_subset, axis=0)  # [A]
                        q_all = np.stack([next_q_list[k][i, j, :] for k in range(self.redq_n)], axis=0)  # [N, A]
                        q_mean = np.mean(q_all, axis=0)  # [A]
                        q_mix = (1.0 - lam) * q_mean + lam * q_min
                        v_next = float(np.max(q_mix))
                    y = _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + gamma_n * v_next
                    action = _action[j][i]
                    for k in range(self.redq_n):
                        final_targets[k][i, j, action] = y

            self.Xs = [_state2_train, _adjs2]
            self.Xs_aux = ([_state2_aux, _adjs2] if _state2_aux is not None else None)
            self.Y_ensemble = final_targets
            self.Y_aux = _aux_target
            self.Y_isr = _isr_target
            self.ssl_action_onehot = _ssl_action_onehot
            self.ssl_next_inputs = _ssl_next_inputs
            self.ssl_sample_weight = self.per_is_weights if getattr(self, "per_is_weights", None) is not None else None
            self.redq_next_state_train = _next_state2_train
            self.redq_action_arr = _action_arr
            self.redq_reward_arr = _reward_arr
            # Keep Y for compatibility with existing logging/guards.
            self.Y = final_targets[0]
            return

        bs = self.dic_agent_conf.get("BATCH_SIZE", 32)
        target = self._q_output_only(self.q_network.predict([_state2_train, _adjs2], batch_size=bs, verbose=0))
        next_state_qvalues = self._q_output_only(
            self.q_network_bar.predict([_next_state2_train, _adjs2], batch_size=bs, verbose=0)
        )

        if self.use_multihead:
            # target: [B, Agents, N, A], next_state_qvalues: [B, Agents, N, A]
            target = np.array(target)
            next_state_qvalues = np.array(next_state_qvalues)
            if self.head_debug:
                print("[MultiHead] target shape:", target.shape,
                      "next_q shape:", next_state_qvalues.shape)
            
            # Phase B: REDQ subset min (with optional mixing)
            if self.use_redq:
                final_target = np.copy(target)
                lam = self.redq_lambda
                for i in range(slice_size):
                    for j in range(self.num_agents):
                        sampled_heads = np.random.choice(self.head_n, self.redq_m, replace=False)
                        sampled_q = next_state_qvalues[i, j, sampled_heads, :]  # [M, A]
                        q_min = np.min(sampled_q, axis=0)  # [A]
                        q_mean = np.mean(next_state_qvalues[i, j], axis=0)  # [A]
                        # Q_mix = (1-λ)*mean(Q) + λ*min(Q_sub)
                        q_mix = (1.0 - lam) * q_mean + lam * q_min
                        v = np.max(q_mix)
                        y = _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                            self.dic_agent_conf["GAMMA"] * v
                        self._assign_multihead_target(final_target, i, j, _action[j][i], y)
            else:
                # Phase A: aggregate over heads for target computation
                next_q_mean = self._aggregate_heads(next_state_qvalues)  # [B, Agents, A]
                final_target = np.copy(target)
                for i in range(slice_size):
                    for j in range(self.num_agents):
                        y = _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                            self.dic_agent_conf["GAMMA"] * np.max(next_q_mean[i, j])
                        self._assign_multihead_target(final_target, i, j, _action[j][i], y)
        else:
            # [batch, agent, num_actions]
            final_target = np.copy(target)
            for i in range(slice_size):
                for j in range(self.num_agents):
                    final_target[i, j, _action[j][i]] = _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                        self.dic_agent_conf["GAMMA"] * np.max(next_state_qvalues[i, j])

        self.Xs = [_state2_train, _adjs2]
        self.Xs_aux = ([_state2_aux, _adjs2] if _state2_aux is not None else None)
        self.Y_aux = _aux_target
        self.Y_isr = _isr_target
        self.Y = final_target
        self.ssl_action_onehot = _ssl_action_onehot
        self.ssl_next_inputs = _ssl_next_inputs
        self.ssl_sample_weight = self.per_is_weights if getattr(self, "per_is_weights", None) is not None else None

    def _assign_multihead_target(self, final_target, b, ag, action, y):
        """
        Assign TD target for multi-head outputs.
        With head bootstrap enabled, only a random subset of heads receives the TD target.
        """
        if self.use_head_bootstrap and self.head_n > 1:
            mask = np.random.rand(self.head_n) < self.head_bootstrap_p
            if not np.any(mask):
                mask[np.random.randint(self.head_n)] = True
            final_target[b, ag, mask, action] = y
            return
        final_target[b, ag, :, action] = y

    def _aggregate_heads(self, head_qvalues):
        """
        Aggregate head dimension for target computation.
        head_qvalues shape: [B, Agents, N, A]
        """
        if self.head_agg == "trimmed_mean":
            # Drop the highest and lowest head for robustness when N>=3.
            n_heads = head_qvalues.shape[2]
            if n_heads >= 3:
                sorted_q = np.sort(head_qvalues, axis=2)
                trimmed = sorted_q[:, :, 1:-1, :]
                return np.mean(trimmed, axis=2)
        return np.mean(head_qvalues, axis=2)

    def _frap_phase_compete_encoder(self, feature, out_dim):
        """
        FRAP-style phase competition encoder adapted to MHQCoSLight:
        movement embedding -> phase embedding -> phase-pair relation-aware competition -> phase scores.
        Input feature: [B, Agents, D], assume D>=20 where [:8]=phase bits, [8:20]=movement features.
        Output: [B, Agents, out_dim] for downstream inter-intersection module.
        """
        if self.len_feature < 20:
            return Dense(out_dim, activation="relu", name="frap_fallback_proj")(feature)

        phase_bits = Lambda(lambda x: x[:, :, :8], name="frap_phase_bits")(feature)
        move_feat = Lambda(lambda x: x[:, :, 8:20], name="frap_move_feat")(feature)

        lane_order = self.dic_traffic_env_conf.get(
            "list_lane_order", ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"]
        )
        move_idx = {"WL": 0, "WT": 1, "EL": 3, "ET": 4, "NL": 6, "NT": 7, "SL": 9, "ST": 10}

        lane_embed = Dense(16, activation="relu", name="frap_lane_embed")
        lane_repr = {}
        for i, lane_name in enumerate(lane_order):
            idx = int(move_idx.get(lane_name, min(i, 11)))
            mv = Lambda(lambda x, j=idx: x[:, :, j:j + 1], name="frap_mv_{}".format(lane_name))(move_feat)
            ph = Lambda(lambda x, j=i: x[:, :, j:j + 1], name="frap_ph_{}".format(lane_name))(phase_bits)
            mv_h = Dense(8, activation="relu", name="frap_mv_proj_{}".format(lane_name))(mv)
            ph_h = Dense(8, activation="relu", name="frap_ph_proj_{}".format(lane_name))(ph)
            lane_repr[lane_name] = lane_embed(
                Concatenate(name="frap_lane_cat_{}".format(lane_name))([mv_h, ph_h])
            )

        phase_list = self.dic_traffic_env_conf.get("PHASE_LIST", ["WT_ET", "NT_ST", "WL_EL", "NL_SL"])
        phase_repr = []
        for ph_name in phase_list:
            if "_" not in ph_name:
                continue
            m1, m2 = ph_name.split("_", 1)
            if m1 in lane_repr and m2 in lane_repr:
                phase_repr.append(Add(name="frap_phase_add_{}".format(ph_name))([lane_repr[m1], lane_repr[m2]]))

        if len(phase_repr) < 2:
            return Dense(out_dim, activation="relu", name="frap_phase_fallback")(feature)

        # [B, Agents, P, d]
        phase_stack = Lambda(lambda xs: tf.stack(xs, axis=2), name="frap_phase_stack")(phase_repr)
        p_num = len(phase_repr)
        rel_dim = 3

        # Build fixed relation one-hot matrix e(p,q): none / partial / full.
        # none: self pair; partial: share one movement; full: no shared movement.
        rel_np = np.zeros((p_num, p_num, rel_dim), dtype=np.float32)
        ph_tokens = []
        for ph_name in phase_list[:p_num]:
            if "_" in ph_name:
                a, b = ph_name.split("_", 1)
                ph_tokens.append((a, b))
            else:
                ph_tokens.append((ph_name, ph_name))
        for i in range(p_num):
            for j in range(p_num):
                if i == j:
                    rel_np[i, j, 0] = 1.0  # none
                else:
                    share = len(set(ph_tokens[i]) & set(ph_tokens[j])) > 0
                    if share:
                        rel_np[i, j, 1] = 1.0  # partial
                    else:
                        rel_np[i, j, 2] = 1.0  # full
        # H_p/H_q: [B, Agents, P, P, d]
        h_p = Lambda(lambda x: tf.tile(tf.expand_dims(x, axis=3), [1, 1, 1, p_num, 1]),
                     name="frap_h_p")(phase_stack)
        h_q = Lambda(lambda x: tf.tile(tf.expand_dims(x, axis=2), [1, 1, p_num, 1, 1]),
                     name="frap_h_q")(phase_stack)

        # Relation embedding H_r: [B, Agents, P, P, d]
        rel_onehot = FRAPRelationTile(rel_matrix=rel_np, name="frap_rel_onehot")(phase_stack)
        h_r = Dense(16, activation="relu", name="frap_rel_embed")(rel_onehot)
        h_r = Dense(16, activation="relu", name="frap_rel_embed2")(h_r)

        # Competition tensor H_c and score matrix C.
        h_c = Lambda(lambda xs: (xs[0] - xs[1]) * xs[2], name="frap_compete")([h_p, h_q, h_r])
        c = Dense(16, activation="relu", name="frap_compete_proj")(h_c)
        c = Dense(1, name="frap_compete_score")(c)
        c = Lambda(lambda x: tf.squeeze(x, axis=-1), name="frap_compete_score_sq")(c)  # [B, Agents, P, P]

        # Phase scores: Q_p = sum_q C_{p,q}
        q_phase = Lambda(lambda x: tf.reduce_sum(x, axis=3), name="frap_q_phase")(c)  # [B, Agents, P]

        # Project phase-score vector back to feature width for downstream modules.
        return Dense(out_dim, activation="relu", name="frap_out_proj")(q_phase)

    def _frap_phase_compete_encoder_strict(self, feature, out_dim, fc_dim=4):
        """
        CoSLight-style FRAP strict structure (state semantics adapted to current project features):
        movement scalar embeddings -> fixed 8-phase construction -> 8x7 phase-pair tensor ->
        1x1 direct conv and relation conv -> fusion conv -> score map -> 64-d feature.
        """
        if self.len_feature < 20:
            return Dense(out_dim, activation="relu", name="frap_strict_fallback")(feature)

        # Current feature layout in this project:
        # [:8] phase bits, [8:20] pressure-like movement signal, [20:32] running-part (if present).
        phase_bits = Lambda(lambda x: x[:, :, :8], name="frs_phase_bits")(feature)
        pressure = Lambda(lambda x: x[:, :, 8:20], name="frs_pressure")(feature)
        if self.len_feature >= 32:
            running = Lambda(lambda x: x[:, :, 20:32], name="frs_running")(feature)
        else:
            running = Lambda(lambda x: tf.zeros_like(x[:, :, 8:20]), name="frs_running_zeros")(feature)

        lane_order = self.dic_traffic_env_conf.get(
            "list_lane_order", ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"]
        )
        move_idx = {"WL": 0, "WT": 1, "EL": 3, "ET": 4, "NL": 6, "NT": 7, "SL": 9, "ST": 10}
        lane_names = lane_order[:8]

        phase_emb = Embedding(input_dim=2, output_dim=fc_dim, name="frs_phase_emb")
        mask_emb = Embedding(input_dim=2, output_dim=fc_dim, name="frs_mask_emb")

        lane_movement_repr = {}
        for i, lane_name in enumerate(lane_names):
            idx = int(move_idx.get(lane_name, min(i, 11)))
            p_scalar = Lambda(lambda x, j=idx: x[:, :, j:j + 1], name="frs_p_{}".format(lane_name))(pressure)
            r_scalar = Lambda(lambda x, j=idx: x[:, :, j:j + 1], name="frs_r_{}".format(lane_name))(running)
            z_scalar = Lambda(lambda x: tf.zeros_like(x), name="frs_z_{}".format(lane_name))(p_scalar)

            car = Dense(fc_dim, activation="sigmoid", name="frs_car_{}".format(lane_name))(p_scalar)
            que = Dense(fc_dim, activation="sigmoid", name="frs_queue_{}".format(lane_name))(p_scalar)
            occ = Dense(fc_dim, activation="sigmoid", name="frs_occ_{}".format(lane_name))(z_scalar)
            flo = Dense(fc_dim, activation="sigmoid", name="frs_flow_{}".format(lane_name))(r_scalar)
            stp = Dense(fc_dim, activation="sigmoid", name="frs_stop_{}".format(lane_name))(r_scalar)

            pha_bin = Lambda(
                lambda x, j=i: tf.cast(tf.round(tf.clip_by_value(x[:, :, j:j + 1], 0.0, 1.0)), tf.int32),
                name="frs_pha_bin_{}".format(lane_name),
            )(phase_bits)
            msk_bin = Lambda(lambda x: tf.ones_like(x, dtype=tf.int32), name="frs_msk_bin_{}".format(lane_name))(pha_bin)
            pha_e = phase_emb(pha_bin)
            msk_e = mask_emb(msk_bin)
            pha = Lambda(lambda x: tf.squeeze(x, axis=2), name="frs_pha_{}".format(lane_name))(pha_e)
            msk = Lambda(lambda x: tf.squeeze(x, axis=2), name="frs_msk_{}".format(lane_name))(msk_e)

            lane_movement_repr[lane_name] = Concatenate(name="frs_mv_cat_{}".format(lane_name))(
                [car, que, occ, flo, stp, pha, msk]
            )  # [B,Agents,7*fc]

        # Fixed CoSLight-style 8 phase construction by lane index in lane_names.
        # Default index pairs follow your provided mix_index convention.
        mix_index = [(3, 7), (6, 7), (2, 3), (2, 6), (1, 5), (4, 5), (0, 1), (0, 4)]
        phase_repr = []
        for p_idx, (a, c) in enumerate(mix_index):
            if a < len(lane_names) and c < len(lane_names):
                phase_repr.append(
                    Add(name="frs_phase_add_{}".format(p_idx))(
                        [lane_movement_repr[lane_names[a]], lane_movement_repr[lane_names[c]]]
                    )
                )
        if len(phase_repr) != 8:
            return Dense(out_dim, activation="relu", name="frap_strict_phase_fallback")(feature)

        # [B,Agents,8,7*fc]
        phase_stack = Lambda(lambda xs: tf.stack(xs, axis=2), name="frs_phase_stack")(phase_repr)
        pair_dim = 14 * fc_dim

        # Build [B,Agents,8,7,14*fc] phase-pair tensor.
        pair_rows = []
        for i in range(8):
            row_pairs = []
            pi = Lambda(lambda x, ii=i: x[:, :, ii, :], name="frs_pi_{}".format(i))(phase_stack)
            for j in range(8):
                if i == j:
                    continue
                pj = Lambda(lambda x, jj=j: x[:, :, jj, :], name="frs_pj_{}_{}".format(i, j))(phase_stack)
                row_pairs.append(Concatenate(name="frs_pair_{}_{}".format(i, j))([pi, pj]))
            row_stack = Lambda(lambda xs: tf.stack(xs, axis=2), name="frs_row_{}".format(i))(row_pairs)  # [B,A,7,C]
            pair_rows.append(row_stack)
        pair_tensor = Lambda(lambda xs: tf.stack(xs, axis=2), name="frs_pair_tensor")(pair_rows)  # [B,A,8,7,C]

        pair_4d = Lambda(
            lambda x, c=pair_dim: tf.reshape(x, [tf.shape(x)[0] * tf.shape(x)[1], 8, 7, c]),
            name="frs_pair_4d",
        )(pair_tensor)
        direct = Conv2D(32, 1, activation="relu", name="frs_direct_conv")(pair_4d)

        # CoSLight binary relation matrix [8,7]: 1 if phase-pair is in conflict style, else 0.
        phase_list = ["WT_ET", "EL_ET", "WL_WT", "WL_EL", "NT_ST", "SL_ST", "NT_NL", "NL_SL"]
        rel_np = []
        for p1 in phase_list:
            row = []
            for p2 in phase_list:
                if p1 == p2:
                    continue
                m1 = p1.split("_")
                m2 = p2.split("_")
                row.append(1 if len(set(m1 + m2)) == 3 else 0)
            rel_np.append(row)
        rel_bin = FRAPBinaryRelationTile(rel_matrix=np.array(rel_np, dtype=np.int32), name="frs_rel_bin")(feature)
        rel_4d = Lambda(
            lambda x: tf.reshape(x, [tf.shape(x)[0] * tf.shape(x)[1], 8, 7]),
            name="frs_rel_4d",
        )(rel_bin)
        rel_onehot = Lambda(lambda x: tf.one_hot(x, depth=2, dtype=tf.float32), name="frs_rel_onehot")(rel_4d)
        rel = Conv2D(32, 1, activation="relu", name="frs_rel_conv")(rel_onehot)

        fused = Multiply(name="frs_mul")([direct, rel])
        fused = Conv2D(16, 1, activation="relu", name="frs_fuse_conv")(fused)
        out_map = Conv2D(8, 1, name="frs_out_conv")(fused)  # [B*A,8,7,8]
        out_map = Lambda(lambda x: tf.reduce_sum(x, axis=2), name="frs_sum7")(out_map)  # [B*A,8,8]
        frap64 = Lambda(
            lambda xs: tf.reshape(xs[0], [tf.shape(xs[1])[0], tf.shape(xs[1])[1], 64]),
            name="frs_out64",
        )([out_map, feature])
        return Dense(out_dim, activation="relu", name="frs_out_proj")(frap64)

    def _build_adaptive_phase_q_head(self, feature_input, feature_hidden):
        """
        Build adaptive pressure phase head for structured Q-network.
        Input: raw feature [B, Agents, D] and MLP output [B, Agents, hidden_dim]
        Output: Q-values [B, Agents, num_actions=4]
        """
        # Assume feature has: phase(8) + pressure(12) + running_part(12) = 32 dims
        # Extract pressure features (movements)
        pressure_features = Lambda(lambda x: x[:, :, 8:20], name="ap_pressure_feat")(feature_input)
        
        # Movement embeddings
        move_embed = Dense(16, activation="relu", name="ap_move_embed")(pressure_features)
        
        # Adaptive pressure per movement (12 movements → 12 scalar pressures)
        adaptive_pressure = Dense(12, kernel_initializer='random_normal', name="ap_adaptive_pressure")(move_embed)
        
        # Phase membership aggregation: simple hardcoded masks for 4-phase intersection
        # Movement order typically: WL, WT, EL, ET, NL, NT, SL, ST, + 4 more features
        # Phases: [WT_ET, NT_ST, WL_EL, NL_SL]
        
        # For simplicity, use fixed phase aggregation:
        # Phase 0 (WT_ET): indices 1, 3
        # Phase 1 (NT_ST): indices 5, 7  
        # Phase 2 (WL_EL): indices 0, 2
        # Phase 3 (NL_SL): indices 4, 6
        
        phase_agg_list = []
        phase_indices = [
            [1, 3],      # Phase 0: WT_ET
            [5, 7],      # Phase 1: NT_ST
            [0, 2],      # Phase 2: WL_EL
            [4, 6],      # Phase 3: NL_SL
        ]
        
        for phase_id, indices in enumerate(phase_indices):
            # Extract and sum adaptive pressure for this phase's movements
            def agg_phase(x, idx_list=indices):
                phase_pressure = tf.add_n([x[:, :, i:i+1] for i in idx_list])  # [B, A, 1]
                return phase_pressure
            
            phase_score = Lambda(agg_phase, name="ap_phase_agg_{}".format(phase_id))(adaptive_pressure)
            phase_agg_list.append(phase_score)
        
        # Concatenate all phase scores: [B, A, 4]
        phase_scores = Concatenate(axis=-1, name="ap_phase_scores")(phase_agg_list)
        
        # Light refinement: normalize by sum for interpretability
        phase_scores_normalized = Lambda(
            lambda x: x / (tf.reduce_sum(x, axis=-1, keepdims=True) + 1e-8),
            name="ap_phase_norm"
        )(phase_scores)
        
        # Combine with hidden representation for final Q values
        q_combined = Concatenate(name="ap_q_combined")([feature_hidden, phase_scores_normalized])
        q_out = Dense(self.num_actions, kernel_initializer='random_normal', name="ap_q_final")(q_combined)
        
        return q_out

    def build_network(self, MLP_layers=None):
        if MLP_layers is None:
            MLP_layers = [self.critic_hidden_dim] * self.critic_num_layers
        CNN_layers = self.CNN_layers
        CNN_heads = [5] * len(CNN_layers)
        In = list()
        # In: [batch,agent,dim]
        # In: [batch,agent,neighbors,agents]
        In.append(Input(shape=(self.num_agents, self.len_feature), name="feature"))
        In.append(Input(shape=(self.num_agents, self.num_neighbors, self.num_agents), name="adjacency_matrix"))

        raw_local_feature = In[0]
        if self.use_feature_group_gate or self.use_feature_group_concat:
            feature = self._build_feature_group_encoder(raw_local_feature, MLP_layers[-1])
        else:
            feature = self.MLP(raw_local_feature, MLP_layers)
        isr_mu = None
        isr_log_var = None
        isr_z = None
        if self.use_isr:
            phase_bits = self._slice_feature_tensor(
                raw_local_feature,
                "cur_phase",
                fallback_dim=8,
                name="isr_phase_bits",
            )
            topology = self._slice_feature_tensor(
                raw_local_feature,
                "intersection_topology_vector",
                fallback_dim=8,
                name="isr_topology",
            )
            isr_enc_in = Concatenate(name="isr_encoder_input")([raw_local_feature, phase_bits, topology])
            isr_h = Dense(
                max(64, MLP_layers[-1]),
                activation="relu",
                kernel_initializer="random_normal",
                name="isr_enc_h1",
            )(isr_enc_in)
            isr_h = Dense(
                max(32, MLP_layers[-1]),
                activation="relu",
                kernel_initializer="random_normal",
                name="isr_enc_h2",
            )(isr_h)
            isr_mu = Dense(
                self.isr_latent_dim,
                kernel_initializer="random_normal",
                name="isr_mu",
            )(isr_h)
            isr_log_var = Dense(
                self.isr_latent_dim,
                kernel_initializer="random_normal",
                name="isr_log_var",
            )(isr_h)

            def _reparameterize(args):
                mu, log_var = args
                eps = K.random_normal(shape=tf.shape(mu))
                sample = mu + tf.exp(0.5 * log_var) * eps
                if self.isr_use_mu_for_acting:
                    return K.in_train_phase(sample, mu)
                return sample

            isr_z = Lambda(_reparameterize, name="isr_latent")([isr_mu, isr_log_var])
            feature = Concatenate(name="isr_fusion_concat")([feature, isr_z])
            feature = Dense(
                MLP_layers[-1],
                activation="relu",
                kernel_initializer="random_normal",
                name="isr_fusion_proj",
            )(feature)
        if self.use_frap_phase_compete:
            if self.use_frap_strict:
                feature = self._frap_phase_compete_encoder_strict(In[0], MLP_layers[-1])
            else:
                feature = self._frap_phase_compete_encoder(In[0], MLP_layers[-1])
        if self.use_intersection_pos_enc and "intersection_topology_vector" in self.feature_slices:
            topo_l, topo_r = self.feature_slices["intersection_topology_vector"]
            topo_vec = Lambda(
                lambda x, l=topo_l, r=topo_r: x[:, :, l:r],
                name="pos_topology_slice",
            )(In[0])
            pos_proj = self._build_intersection_positional_encoding(topo_vec, MLP_layers[-1])
            feature = Add(name="intersection_pos_fused_residual")([feature, pos_proj])
        att_adj = In[1]
        candidate_adj = In[1]
        cos_probs = None
        relation_debug = None
        if self.cos_enabled:
            if self.use_dynamic_collab_full:
                cos_logits = self._build_dynamic_collab_full_logits(feature, raw_local_feature)
            else:
                cos_logits = Dense(
                    self.num_agents,
                    kernel_initializer='random_normal',
                    name='cos_logits'
                )(feature)
            if self.cos_use_input_candidate_mask:
                cos_logits = Lambda(
                    self._mask_scores_with_candidate_adj,
                    name="cos_logits_masked"
                )([cos_logits, candidate_adj])
            cos_probs = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name="cos_probs")(cos_logits)
            att_adj = CoSDynamicAdjacency(
                num_agents=self.num_agents,
                total_k=self.cos_select_k,
                include_self=self.cos_include_self,
                adj_mode=self.cos_adj_mode,
                slot_min_prob=self.cos_slot_min_prob,
                slot_budget_tau=self.cos_budget_tau,
                explore_mode=self.cos_explore_mode,
                explore_prob=self.cos_explore_prob,
                gumbel_tau=self.cos_gumbel_tau,
                gumbel_scale=self.cos_gumbel_scale,
                explore_infer=self.cos_explore_infer,
                name="cos_dynamic_adj"
            )([cos_logits, candidate_adj] if self.cos_use_input_candidate_mask else cos_logits)
        if self.use_transformer_encoder:
            trans_dim = self.trans_dim if self.trans_dim > 0 else MLP_layers[-1]
            if trans_dim != MLP_layers[-1]:
                h = Dense(
                    trans_dim,
                    activation="relu",
                    kernel_initializer="random_normal",
                    name="trans_input_proj",
                )(feature)
            else:
                h = feature

            if self.trans_use_cos_mask:
                attn_mask = Lambda(self._adjacency_to_attn_mask, name="trans_attn_mask")(att_adj)
            else:
                attn_mask = Lambda(self._full_attn_mask_from_feature, name="trans_attn_mask_full")(h)
            h = self._transformer_encoder_stack(h, attn_mask, trans_dim)
        else:
            if self.use_gat_agg:
                # feature:[batch,agents,feature_dim]
                print("CNN_heads:", CNN_heads)
                att_fn = self.CompetitiveAttModel if self.use_competitive_agg else self.MultiHeadsAttModel
                for CNN_layer_index, CNN_layer_size in enumerate(CNN_layers):
                    print("CNN_heads[CNN_layer_index]:", CNN_heads[CNN_layer_index])
                    if CNN_layer_index == 0:
                        h, _ = att_fn(
                            feature,
                            att_adj,
                            d_in=MLP_layers[-1],
                            h_dim=CNN_layer_size[0],
                            dout=CNN_layer_size[1],
                            head=CNN_heads[CNN_layer_index],
                            suffix=CNN_layer_index
                        )
                    else:
                        h, _ = att_fn(
                            h,
                            att_adj,
                            d_in=MLP_layers[-1],
                            h_dim=CNN_layer_size[0],
                            dout=CNN_layer_size[1],
                            head=CNN_heads[CNN_layer_index],
                            suffix=CNN_layer_index
                        )
            else:
                # MLP-only ablation: no Transformer and no GAT aggregation.
                h = feature
                if self.use_neighbor_h_mean_concat:
                    h = self._build_neighbor_h_mean_concat(
                        h=h,
                        candidate_adj=In[1],
                        name_prefix="neighbor_h_mean",
                    )
                elif self.use_delay_msg_mean or self.use_delay_rel_msg_mean:
                    h = self._build_delay_message_mean_agg(
                        h=h,
                        raw_local_feature=raw_local_feature,
                        candidate_adj=In[1],
                        out_dim=MLP_layers[-1],
                        use_rel=self.use_delay_rel_msg_mean,
                        name_prefix="delay_rel_msg_mean" if self.use_delay_rel_msg_mean else "delay_msg_mean",
                    )
                elif self.use_mlp_neighbor_agg:
                    # For dynamic collaboration, use the learned top-k collaborator set
                    # as the primary aggregation support, then keep soft weights only
                    # inside the selected subset. This makes collaborator selection
                    # behave like a real sparse collaborator picker instead of a
                    # full-graph soft attention layer.
                    if self.cos_enabled and cos_probs is not None:
                        nei_weights = Lambda(
                            self._cos_probs_to_topk_row_norm_weights,
                            name="mlp_neighbor_weights"
                        )([cos_probs, att_adj])
                    else:
                        nei_weights = Lambda(
                            self._adjacency_to_row_norm_weights,
                            name="mlp_neighbor_weights"
                        )(att_adj)
                    h_agg = Lambda(
                        lambda xs: tf.matmul(xs[0], xs[1]),
                        name="mlp_neighbor_agg"
                    )([nei_weights, h])
                    h_res = Dense(
                        MLP_layers[-1],
                        use_bias=False,
                        kernel_initializer="random_normal",
                        name="mlp_neighbor_proj"
                    )(h_agg)
                    h = Add(name="mlp_neighbor_residual")([h, h_res])
                relation_debug = None
            if self.use_transformer_encoder or self.use_gat_agg:
                relation_debug = None
        h = Lambda(lambda x: x, name="latent_repr")(h)
        # action prediction layer
        if self.use_adaptive_pressure_phase_head:
            out = self._build_adaptive_phase_q_head(In[0], h)
            print("[APLight] structured phase-pressure head, output shape: [B, {}, {}]".format(
                self.num_agents, self.num_actions
            ))
        elif self.use_multihead:
            # N independent heads: each [B, Agents, A], stacked → [B, Agents, N, A]
            heads = [self._dense_or_noisy(
                self.num_actions,
                kernel_initializer='random_normal',
                name='q_head_{}'.format(k),
            )(h) for k in range(self.head_n)]
            out = StackHeads(name='stack_heads')(heads)
            print("[MultiHead] output shape: [B, {}, {}, {}]".format(
                self.num_agents, self.head_n, self.num_actions))
        elif self.use_dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
            # V stream: [B, Agents, 1]
            v_stream = self._dense_or_noisy(1, kernel_initializer='random_normal', name='dueling_value')(h)
            # A stream: [B, Agents, num_actions]
            a_stream = self._dense_or_noisy(self.num_actions, kernel_initializer='random_normal', name='dueling_advantage')(h)
            # Combine: Q = V + A - mean(A), keeping mean centered
            out = Lambda(
                lambda x: x[0] + x[1] - tf.reduce_mean(x[1], axis=-1, keepdims=True),
                name='dueling_combine'
            )([v_stream, a_stream])
            print("[Dueling] V+A architecture, output shape: [B, {}, {}]".format(
                self.num_agents, self.num_actions))
        else:
            out = self._dense_or_noisy(self.num_actions, kernel_initializer='random_normal', name='action_layer')(h)
        outputs = out
        isr_out = None
        if self.use_auxiliary_head and self._auxiliary_target_dim() > 0:
            aux_out = Dense(
                self._auxiliary_target_dim(),
                kernel_initializer='random_normal',
                name='auxiliary_head'
            )(h)
            outputs = [out, aux_out]
        if self.use_isr and isr_z is not None:
            isr_decoder_in = Concatenate(name="isr_decoder_input")([h, isr_z, raw_local_feature])
            isr_dec = Dense(
                max(64, self.len_feature),
                activation="relu",
                kernel_initializer="random_normal",
                name="isr_dec_h1",
            )(isr_decoder_in)
            isr_dec = Dense(
                max(32, self.len_feature // 2),
                activation="relu",
                kernel_initializer="random_normal",
                name="isr_dec_h2",
            )(isr_dec)
            isr_out = Dense(
                self._isr_target_dim(),
                kernel_initializer="random_normal",
                name="isr_recon_head",
            )(isr_dec)
            outputs = [out] if not isinstance(outputs, list) else list(outputs)
            outputs.append(isr_out)
        # out:[batch,agent,action] or [batch,agent,N,action]
        model = Model(inputs=In, outputs=outputs)
        if relation_debug is not None:
            model.add_metric(tf.reduce_mean(relation_debug["k_eff"]), name="neighbor_keff_mean")
            model.add_metric(tf.reduce_min(relation_debug["k_eff"]), name="neighbor_keff_min")
            model.add_metric(tf.reduce_max(relation_debug["k_eff"]), name="neighbor_keff_max")
            model.add_metric(tf.reduce_mean(relation_debug["topk_probs"]), name="neighbor_topk_p_mean")
            model.add_metric(tf.reduce_min(relation_debug["topk_probs"]), name="neighbor_topk_p_min")
            model.add_metric(tf.reduce_max(relation_debug["topk_probs"]), name="neighbor_topk_p_max")
            model.add_metric(tf.reduce_mean(relation_debug["gate"]), name="neighbor_gate_mean")
            model.add_metric(tf.reduce_min(relation_debug["gate"]), name="neighbor_gate_min")
            model.add_metric(tf.reduce_max(relation_debug["gate"]), name="neighbor_gate_max")

        if self.cos_enabled and cos_probs is not None:
            diag = tf.linalg.diag_part(cos_probs)
            diag_loss = -tf.reduce_mean(diag)
            sym_loss = tf.reduce_mean(tf.square(cos_probs - tf.transpose(cos_probs, perm=[0, 2, 1])))
            entropy = -tf.reduce_mean(tf.reduce_sum(cos_probs * tf.math.log(cos_probs + 1e-8), axis=-1))

            # Temporal smoothness proxy: penalize changes between consecutive items in the batch.
            # NOTE: This assumes batches preserve temporal order (we already fit(shuffle=False)).
            temporal_loss = tf.constant(0.0, dtype=cos_probs.dtype)
            if self.cos_temporal_smooth_coef > 0:
                diff = cos_probs[1:] - cos_probs[:-1]
                sq = tf.square(diff)
                # Use reduce_mean instead of tf.size-based normalization.
                # tf.size can become fragile after model JSON round-trips in TF/Keras.
                temporal_loss = tf.reduce_mean(sq)

            # Budget / adaptive-K proxy: encourage sparsity by penalizing the expected number
            # of edges whose probability exceeds a threshold (smooth indicator).
            budget_loss = tf.constant(0.0, dtype=cos_probs.dtype)
            budget_k_mean = tf.constant(0.0, dtype=cos_probs.dtype)
            if self.cos_budget_coef > 0 and self.cos_budget_thr > 0:
                n = self.num_agents
                eye = tf.eye(n, dtype=cos_probs.dtype)
                eye_b = tf.tile(tf.reshape(eye, [1, n, n]), [tf.shape(cos_probs)[0], 1, 1])
                thr = tf.constant(self.cos_budget_thr, dtype=cos_probs.dtype)
                tau = tf.constant(max(self.cos_budget_tau, 1e-6), dtype=cos_probs.dtype)
                active = tf.sigmoid((cos_probs - thr) / tau) * (1.0 - eye_b)
                budget_k = tf.reduce_sum(active, axis=-1)  # [B,N]
                budget_k_mean = tf.reduce_mean(budget_k)
                budget_loss = budget_k_mean

            if self.cos_beta_diag > 0:
                model.add_loss(self.cos_beta_diag * diag_loss)
            if self.cos_gamma_sym > 0:
                model.add_loss(self.cos_gamma_sym * sym_loss)
            if self.cos_entropy_coef > 0:
                model.add_loss(-self.cos_entropy_coef * entropy)
            if self.cos_temporal_smooth_coef > 0:
                model.add_loss(self.cos_temporal_smooth_coef * temporal_loss)
            if self.cos_budget_coef > 0 and self.cos_budget_thr > 0:
                model.add_loss(self.cos_budget_coef * budget_loss)
            model.add_metric(diag_loss, name="cos_diag_loss", aggregation="mean")
            model.add_metric(sym_loss, name="cos_sym_loss", aggregation="mean")
            model.add_metric(entropy, name="cos_entropy", aggregation="mean")
            if self.cos_temporal_smooth_coef > 0:
                model.add_metric(temporal_loss, name="cos_temporal_smooth", aggregation="mean")
            if self.cos_budget_coef > 0 and self.cos_budget_thr > 0:
                model.add_metric(budget_k_mean, name="cos_budget_k_mean", aggregation="mean")

        if self.use_isr and isr_mu is not None and isr_log_var is not None:
            kl_loss = 0.5 * tf.reduce_mean(
                tf.exp(isr_log_var) + tf.square(isr_mu) - 1.0 - isr_log_var
            )
            centers = tf.reduce_mean(isr_mu, axis=0)  # [Agents, latent]
            within = tf.reduce_mean(tf.square(isr_mu - tf.expand_dims(centers, axis=0)))
            pairwise = tf.reduce_sum(
                tf.square(tf.expand_dims(centers, axis=1) - tf.expand_dims(centers, axis=0)),
                axis=-1,
            )
            n_agents = tf.shape(pairwise)[0]
            offdiag_mask = tf.ones_like(pairwise) - tf.eye(n_agents, dtype=pairwise.dtype)
            sep_pen = tf.reduce_sum(tf.nn.relu(self.isr_contrastive_margin - pairwise) * offdiag_mask) / (
                tf.reduce_sum(offdiag_mask) + 1e-6
            )
            contrastive_loss = within + sep_pen
            if self.isr_beta_kl > 0:
                model.add_loss(self.isr_beta_kl * kl_loss)
            if self.isr_contrastive_weight > 0:
                model.add_loss(self.isr_contrastive_weight * contrastive_loss)
            model.add_metric(kl_loss, name="isr_kl", aggregation="mean")
            model.add_metric(within, name="isr_within", aggregation="mean")
            model.add_metric(sep_pen, name="isr_sep_pen", aggregation="mean")
            model.add_metric(tf.reduce_mean(tf.norm(isr_mu, axis=-1)), name="isr_mu_norm", aggregation="mean")

        if self.use_auxiliary_head and self._auxiliary_target_dim() > 0 and self.use_isr:
            model.compile(
                optimizer=Adam(lr=self.dic_agent_conf.get("LEARNING_RATE", 0.0005)),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"], "mse", "mse"],
                loss_weights=[1.0, self.auxiliary_weight, self.isr_recon_weight],
            )
        elif self.use_auxiliary_head and self._auxiliary_target_dim() > 0:
            model.compile(
                optimizer=Adam(lr=self.dic_agent_conf.get("LEARNING_RATE", 0.0005)),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"], "mse"],
                loss_weights=[1.0, self.auxiliary_weight],
            )
        elif self.use_isr:
            model.compile(
                optimizer=Adam(lr=self.dic_agent_conf.get("LEARNING_RATE", 0.0005)),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"], "mse"],
                loss_weights=[1.0, self.isr_recon_weight],
            )
        else:
            model.compile(optimizer=Adam(lr=self.dic_agent_conf.get("LEARNING_RATE", 0.0005)),
                          loss=self.dic_agent_conf["LOSS_FUNCTION"])
        model.summary()
        return model

    @staticmethod
    def _weighted_mean(per_item_loss, sample_weight=None):
        if sample_weight is None:
            return tf.reduce_mean(per_item_loss)
        sw = tf.cast(sample_weight, per_item_loss.dtype)
        if len(sw.shape) == 1:
            sw = tf.expand_dims(sw, axis=-1)
        numer = tf.reduce_sum(per_item_loss * sw)
        denom = tf.reduce_sum(sw) + tf.constant(1e-6, dtype=per_item_loss.dtype)
        return numer / denom

    def _predict_q_tensor(self, net, state, adj, training=False):
        pred = net([state, adj], training=training)
        return tf.cast(self._q_output_only(pred), tf.float32)

    def _build_true_redq_targets_from_q(
        self,
        q_next_online_list,
        q_next_target_list,
        action_arr,
        reward_arr,
    ):
        batch_n = int(action_arr.shape[0])
        gamma_n = float(self.dic_agent_conf["GAMMA"] ** self.nstep)
        lam = float(self.redq_lambda)
        normal_factor = float(self.dic_agent_conf["NORMAL_FACTOR"])

        y = np.zeros((batch_n, self.num_agents), dtype=np.float32)
        for i in range(batch_n):
            for j in range(self.num_agents):
                sampled = self._sample_redq_indices()
                if self.use_double_dqn:
                    q_all_online = np.stack([q_next_online_list[k][i, j, :] for k in range(self.redq_n)], axis=0)
                    q_mean_online = np.mean(q_all_online, axis=0)
                    q_sub_online = np.stack([q_next_online_list[k][i, j, :] for k in sampled], axis=0)
                    q_min_online = np.min(q_sub_online, axis=0)
                    q_mix_online = (1.0 - lam) * q_mean_online + lam * q_min_online
                    best_a = int(np.argmax(q_mix_online))

                    q_all_tgt = np.stack([q_next_target_list[k][i, j, best_a] for k in range(self.redq_n)], axis=0)
                    q_mean_tgt = float(np.mean(q_all_tgt))
                    q_sub_tgt = np.stack([q_next_target_list[k][i, j, best_a] for k in sampled], axis=0)
                    q_min_tgt = float(np.min(q_sub_tgt))
                    v_next = (1.0 - lam) * q_mean_tgt + lam * q_min_tgt
                else:
                    q_subset = np.stack([q_next_target_list[k][i, j, :] for k in sampled], axis=0)
                    q_min = np.min(q_subset, axis=0)
                    q_all = np.stack([q_next_target_list[k][i, j, :] for k in range(self.redq_n)], axis=0)
                    q_mean = np.mean(q_all, axis=0)
                    q_mix = (1.0 - lam) * q_mean + lam * q_min
                    v_next = float(np.max(q_mix))

                y[i, j] = reward_arr[i, j] / normal_factor + gamma_n * v_next
        return y

    def _crossq_safe_true_redq_train_batch(
        self,
        state_batch,
        adj_batch,
        next_state_batch,
        action_arr,
        reward_arr,
        sample_weight=None,
    ):
        state = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        adj = tf.convert_to_tensor(adj_batch, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        action = tf.convert_to_tensor(action_arr, dtype=tf.int32)
        reward = np.asarray(reward_arr, dtype=np.float32)
        sample_weight_tf = None
        if sample_weight is not None:
            sample_weight_tf = tf.convert_to_tensor(sample_weight, dtype=tf.float32)

        q_cur_list = []
        q_next_online_list = []
        tapes = []
        joint_state = tf.concat([state, next_state], axis=0) if self.crossq_joint_forward else None
        joint_adj = tf.concat([adj, adj], axis=0) if self.crossq_joint_forward else None
        batch_n = int(state_batch.shape[0])

        for net in self.q_ensemble:
            tape = tf.GradientTape()
            tape.__enter__()
            if self.crossq_joint_forward:
                q_joint = self._predict_q_tensor(net, joint_state, joint_adj, training=True)
                q_cur, q_next_online = tf.split(q_joint, [batch_n, batch_n], axis=0)
            else:
                q_cur = self._predict_q_tensor(net, state, adj, training=True)
                q_next_online = self._predict_q_tensor(net, next_state, adj, training=True)
            q_cur_list.append(q_cur)
            q_next_online_list.append(tf.stop_gradient(q_next_online))
            tapes.append(tape)

        target_adj = adj
        q_next_target_list = [
            self._predict_q_tensor(net_bar, next_state, target_adj, training=False)
            for net_bar in self.q_ensemble_bar
        ]

        q_next_online_np = [q.numpy() for q in q_next_online_list]
        q_next_target_np = [q.numpy() for q in q_next_target_list]
        target_y = tf.convert_to_tensor(
            self._build_true_redq_targets_from_q(
                q_next_online_np,
                q_next_target_np,
                np.asarray(action_arr, dtype=np.int32),
                reward,
            ),
            dtype=tf.float32,
        )
        target_y = tf.stop_gradient(target_y)

        action_onehot = tf.one_hot(action, depth=self.num_actions, dtype=tf.float32)
        losses = []
        for q_idx, net in enumerate(self.q_ensemble):
            q_selected = tf.reduce_sum(q_cur_list[q_idx] * action_onehot, axis=-1)
            per_item_loss = tf.square(target_y - q_selected)
            loss = self._weighted_mean(per_item_loss, sample_weight=sample_weight_tf)
            gradients = tapes[q_idx].gradient(loss, net.trainable_variables)
            tapes[q_idx].__exit__(None, None, None)
            grad_vars = [
                (grad, var)
                for grad, var in zip(gradients, net.trainable_variables)
                if grad is not None
            ]
            if grad_vars:
                net.optimizer.apply_gradients(grad_vars)
            losses.append(float(loss.numpy()))

        if bool(self.dic_agent_conf.get("REDQ_SOFT_TARGET_UPDATE", False)):
            self._soft_update_true_redq_targets()
        self.q_network = self.q_ensemble[0]
        self.q_network_bar = self.q_ensemble_bar[0]
        return losses

    def train_network(self):
        if self.use_true_redq_ensemble:
            if (not hasattr(self, "Y_ensemble")) or self.Y_ensemble is None or len(self.Y_ensemble) == 0:
                return
            if (
                getattr(self, "redq_next_state_train", None) is None
                or getattr(self, "redq_action_arr", None) is None
                or getattr(self, "redq_reward_arr", None) is None
            ):
                return
            epochs = self.dic_agent_conf["EPOCHS"]
            batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y_ensemble[0]))
            use_critic_bootstrap_sample = bool(self.dic_agent_conf.get("USE_CRITIC_BOOTSTRAP_SAMPLE", True))
            utd = max(1, int(self.dic_agent_conf.get("REDQ_UTD", 1)))
            
            # Get PER IS weights if available
            sample_weight = None
            if hasattr(self, 'per_is_weights') and self.per_is_weights is not None:
                sample_weight = self.per_is_weights

            if self.crossq_safe_mode and self.crossq_custom_train_step:
                sample_n = len(self.Y_ensemble[0])
                for utd_step in range(utd):
                    if batch_size >= sample_n:
                        batch_idx = np.arange(sample_n)
                    else:
                        batch_idx = np.random.choice(sample_n, size=batch_size, replace=False)
                    Xs_q = [x[batch_idx] for x in self.Xs]
                    next_state_q = self.redq_next_state_train[batch_idx]
                    action_q = self.redq_action_arr[batch_idx]
                    reward_q = self.redq_reward_arr[batch_idx]
                    sw_q = sample_weight[batch_idx] if sample_weight is not None else None
                    losses = self._crossq_safe_true_redq_train_batch(
                        Xs_q[0],
                        Xs_q[1],
                        next_state_q,
                        action_q,
                        reward_q,
                        sample_weight=sw_q,
                    )
                    if utd_step == 0 and losses:
                        print(
                            "[CrossQ-Safe] first critic losses: mean={:.6f}, first={:.6f}".format(
                                float(np.mean(losses)),
                                float(losses[0]),
                            )
                        )
                self._refresh_cos_prob_model()
                return

            if self.use_official_redq_update:
                sample_n = len(self.Y_ensemble[0])
                for utd_step in range(utd):
                    if batch_size >= sample_n:
                        batch_idx = np.arange(sample_n)
                    else:
                        batch_idx = np.random.choice(sample_n, size=batch_size, replace=False)
                    Xs_q = [x[batch_idx] for x in self.Xs]
                    next_state_q = self.redq_next_state_train[batch_idx]
                    action_q = self.redq_action_arr[batch_idx]
                    reward_q = self.redq_reward_arr[batch_idx]
                    Xs_aux_q = [x[batch_idx] for x in self.Xs_aux] if self.Xs_aux is not None else None
                    sw_q = sample_weight[batch_idx] if sample_weight is not None else None
                    Y_aux_q = self.Y_aux[batch_idx] if self.Y_aux is not None else None
                    Y_isr_q = self.Y_isr[batch_idx] if self.Y_isr is not None else None
                    ssl_action_q = self.ssl_action_onehot[batch_idx] if self.ssl_action_onehot is not None else None
                    ssl_next_q = [x[batch_idx] for x in self.ssl_next_inputs] if self.ssl_next_inputs is not None else None
                    Y_batch = self._build_true_redq_targets_batch(
                        Xs_q[0],
                        Xs_q[1],
                        next_state_q,
                        action_q,
                        reward_q,
                    )
                    for q_idx, net in enumerate(self.q_ensemble):
                        Y_q = Y_batch[q_idx]
                        targets = self._pack_model_targets(Y_q, Y_aux_q, Y_isr_q)
                        sw_list = self._pack_model_sample_weight(sw_q, Y_aux_q, Y_isr_q)
                        loss = net.train_on_batch(Xs_q, targets, sample_weight=sw_list)
                        self._maybe_train_q_consistency_aux(net, Xs_q, Xs_aux_q)
                        if q_idx == 0 and utd_step == 0:
                            try:
                                print("[Official-REDQ] first train_on_batch loss:", float(loss))
                            except Exception:
                                print("[Official-REDQ] first train_on_batch loss:", loss)
                        self._train_latent_transition_ssl(q_idx, Xs_q, ssl_next_q, ssl_action_q, sample_weight=sw_q)
                    self._soft_update_true_redq_targets()
                self.q_network = self.q_ensemble[0]
                self._refresh_cos_prob_model()
                return
            
            for utd_step in range(utd):
                Y_batch_all = None
                if not use_critic_bootstrap_sample:
                    Y_batch_all = self._build_true_redq_targets_batch(
                        self.Xs[0],
                        self.Xs[1],
                        self.redq_next_state_train,
                        self.redq_action_arr,
                        self.redq_reward_arr,
                    )
                for q_idx, net in enumerate(self.q_ensemble):
                    if use_critic_bootstrap_sample:
                        # Each critic trains on its own bootstrap-resampled minibatch.
                        sample_n = len(self.Y_ensemble[q_idx])
                        bootstrap_idx = np.random.randint(0, sample_n, size=sample_n)
                        Xs_q = [x[bootstrap_idx] for x in self.Xs]
                        next_state_q = self.redq_next_state_train[bootstrap_idx]
                        action_q = self.redq_action_arr[bootstrap_idx]
                        reward_q = self.redq_reward_arr[bootstrap_idx]
                        Xs_aux_q = [x[bootstrap_idx] for x in self.Xs_aux] if self.Xs_aux is not None else None
                        Y_aux_q = self.Y_aux[bootstrap_idx] if self.Y_aux is not None else None
                        Y_isr_q = self.Y_isr[bootstrap_idx] if self.Y_isr is not None else None
                        # Resample weights accordingly if PER is enabled
                        sw_q = sample_weight[bootstrap_idx] if sample_weight is not None else None
                        ssl_action_q = self.ssl_action_onehot[bootstrap_idx] if self.ssl_action_onehot is not None else None
                        ssl_next_q = [x[bootstrap_idx] for x in self.ssl_next_inputs] if self.ssl_next_inputs is not None else None
                        Y_q = self._build_true_redq_targets_batch(
                            Xs_q[0],
                            Xs_q[1],
                            next_state_q,
                            action_q,
                            reward_q,
                        )[q_idx]
                    else:
                        Xs_q = self.Xs
                        next_state_q = self.redq_next_state_train
                        action_q = self.redq_action_arr
                        reward_q = self.redq_reward_arr
                        Xs_aux_q = self.Xs_aux
                        Y_aux_q = self.Y_aux
                        Y_isr_q = self.Y_isr
                        sw_q = sample_weight
                        ssl_action_q = self.ssl_action_onehot
                        ssl_next_q = self.ssl_next_inputs
                        Y_q = Y_batch_all[q_idx]
                    targets = self._pack_model_targets(Y_q, Y_aux_q, Y_isr_q)
                    sw_list = self._pack_model_sample_weight(sw_q, Y_aux_q, Y_isr_q)
                    self._train_model_with_batches(
                        net=net,
                        inputs=Xs_q,
                        targets=targets,
                        sample_weight=sw_list,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=(q_idx == 0 and utd_step == 0),
                    )
                    self._maybe_train_q_consistency_aux(net, Xs_q, Xs_aux_q)
                    self._train_latent_transition_ssl(q_idx, Xs_q, ssl_next_q, ssl_action_q, sample_weight=sw_q)
                self._soft_update_true_redq_targets()
            self.q_network = self.q_ensemble[0]
            self._refresh_cos_prob_model()
            return

        if not hasattr(self, "Y") or self.Y is None or len(self.Y) == 0:
            return
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))
        utd = max(1, int(self.dic_agent_conf.get("REDQ_UTD", 1)))
        
        # Get PER IS weights if available
        sample_weight = None
        if hasattr(self, 'per_is_weights') and self.per_is_weights is not None:
            sample_weight = self.per_is_weights

        sample_n = len(self.Y)
        for utd_step in range(utd):
            if batch_size >= sample_n:
                batch_idx = np.arange(sample_n)
            else:
                batch_idx = np.random.choice(sample_n, size=batch_size, replace=False)
            Xs_q = [x[batch_idx] for x in self.Xs]
            Y_q = self.Y[batch_idx]
            Y_aux_q = self.Y_aux[batch_idx] if self.Y_aux is not None else None
            Y_isr_q = self.Y_isr[batch_idx] if self.Y_isr is not None else None
            sw_q = sample_weight[batch_idx] if sample_weight is not None else None
            targets = self._pack_model_targets(Y_q, Y_aux_q, Y_isr_q)
            sw_list = self._pack_model_sample_weight(sw_q, Y_aux_q, Y_isr_q)
            self._train_model_with_batches(
                net=self.q_network,
                inputs=Xs_q,
                targets=targets,
                sample_weight=sw_list,
                batch_size=batch_size,
                epochs=epochs,
                verbose=(utd_step == 0),
            )
        self._train_latent_transition_ssl(0, self.Xs, self.ssl_next_inputs, self.ssl_action_onehot, sample_weight=sample_weight)

    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network = self.build_network()
        network.set_weights(network_copy.get_weights())
        if self.use_auxiliary_head and self._auxiliary_target_dim() > 0 and self.use_isr:
            network.compile(
                optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"], "mse", "mse"],
                loss_weights=[1.0, self.auxiliary_weight, self.isr_recon_weight],
            )
        elif self.use_auxiliary_head and self._auxiliary_target_dim() > 0:
            network.compile(
                optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"], "mse"],
                loss_weights=[1.0, self.auxiliary_weight],
            )
        elif self.use_isr:
            network.compile(
                optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"], "mse"],
                loss_weights=[1.0, self.isr_recon_weight],
            )
        else:
            network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                            loss=self.dic_agent_conf["LOSS_FUNCTION"])

        return network

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        if self.use_true_redq_ensemble:
            self.q_ensemble = []
            for q_idx in range(self.redq_n):
                path = os.path.join(
                    file_path,
                    "{}.h5".format(self._ensemble_file_name(file_name, q_idx)),
                )
                net = self.build_network()
                net.load_weights(path)
                self.q_ensemble.append(net)
            self.q_network = self.q_ensemble[0]
        else:
            custom_objs = self._custom_objects()
            self.q_network = load_model(
                os.path.join(file_path, "%s.h5" % file_name),
                custom_objects=custom_objs)
        self._refresh_cos_prob_model()
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        if self.use_true_redq_ensemble:
            self.q_ensemble_bar = []
            for q_idx in range(self.redq_n):
                path = os.path.join(
                    file_path,
                    "{}.h5".format(self._ensemble_file_name(file_name, q_idx)),
                )
                net = self.build_network()
                net.load_weights(path)
                self.q_ensemble_bar.append(net)
            self.q_network_bar = self.q_ensemble_bar[0]
        else:
            custom_objs = self._custom_objects()
            self.q_network_bar = load_model(
                os.path.join(file_path, "%s.h5" % file_name),
                custom_objects=custom_objs)
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        if self.use_true_redq_ensemble:
            for q_idx, net in enumerate(self.q_ensemble):
                final_path = os.path.join(
                    self.dic_path["PATH_TO_MODEL"],
                    "{}.h5".format(self._ensemble_file_name(file_name, q_idx)),
                )
                self._atomic_save_weights(net, final_path)
            return
        final_path = os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name)
        self._atomic_save_model(self.q_network, final_path)

    def save_network_bar(self, file_name):
        if self.use_true_redq_ensemble:
            for q_idx, net in enumerate(self.q_ensemble_bar):
                final_path = os.path.join(
                    self.dic_path["PATH_TO_MODEL"],
                    "{}.h5".format(self._ensemble_file_name(file_name, q_idx)),
                )
                self._atomic_save_weights(net, final_path)
            return
        final_path = os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name)
        self._atomic_save_model(self.q_network_bar, final_path)

    def _atomic_save_weights(self, net, final_path):
        model_dir = os.path.dirname(final_path)
        os.makedirs(model_dir, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_weights_",
            suffix=".h5",
            dir=model_dir,
        )
        os.close(tmp_fd)
        try:
            net.save_weights(tmp_path)
            os.replace(tmp_path, final_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _atomic_save_model(self, net, final_path):
        model_dir = os.path.dirname(final_path)
        os.makedirs(model_dir, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_model_",
            suffix=".h5",
            dir=model_dir,
        )
        os.close(tmp_fd)
        try:
            os.remove(tmp_path)
            net.save(tmp_path)
            os.replace(tmp_path, final_path)
        finally:
            if os.path.exists(tmp_path):
                if os.path.isdir(tmp_path):
                    shutil.rmtree(tmp_path, ignore_errors=True)
                else:
                    os.remove(tmp_path)


class RepeatVector3D(Layer):
    def __init__(self, times, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def call(self, inputs):
        # [batch,agent,dim]->[batch,1,agent,dim]
        # [batch,1,agent,dim]->[batch,agent,agent,dim]
        return K.tile(K.expand_dims(inputs, 1), [1, self.times, 1, 1])

    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
