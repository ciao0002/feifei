"""
MHQ-CoSLight agent.
observations: [lane_num_vehicle, cur_phase]
reward: -queue_length
"""
import numpy as np
import os
from .agent import Agent
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Dense,
    Dropout,
    Lambda,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
)
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


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


class CoSDynamicAdjacency(Layer):
    """
    Build dynamic collaborator adjacency from CoS logits/probabilities.
    Output shape: [B, N, K, N], where K includes self when include_self=True.
    """
    def __init__(self, num_agents, total_k, include_self=True, **kwargs):
        super(CoSDynamicAdjacency, self).__init__(**kwargs)
        self.num_agents = int(num_agents)
        self.total_k = int(total_k)
        self.include_self = bool(include_self)

    def call(self, scores):
        # scores: [B, N, N] (logits)
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
            topk = tf.math.top_k(probs_others, k=other_k, sorted=False)
            mask = tf.reduce_sum(tf.one_hot(topk.indices, depth=n, dtype=dtype), axis=2)  # [B,N,N]
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
        self.cos_include_self = bool(dic_agent_conf.get("COS_INCLUDE_SELF", True))
        self.cos_beta_diag = float(dic_agent_conf.get("COS_BETA_DIAG", 0.0))
        self.cos_gamma_sym = float(dic_agent_conf.get("COS_GAMMA_SYM", 0.0))
        self.cos_entropy_coef = float(dic_agent_conf.get("COS_ENTROPY_COEF", 0.0))
        # Keep semantic consistency: K refers to collaborator count (including self by default).
        if self.cos_enabled:
            self.num_neighbors = min(self.cos_total_k, self.num_agents)
        else:
            self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()
        self.memory = build_memory()

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
        
        # Phase B: REDQ
        self.use_redq = dic_agent_conf.get("USE_REDQ", False)
        self.redq_m = dic_agent_conf.get("REDQ_M", 2)
        self.redq_lambda = dic_agent_conf.get("REDQ_LAMBDA", 1.0)

        # CityLight-inspired: competitive neighbor aggregation
        self.use_competitive_agg = dic_agent_conf.get("USE_COMPETITIVE_AGG", False)
        # Optional CoSLight-style Transformer encoder (default OFF to keep behavior unchanged).
        self.use_transformer_encoder = bool(dic_agent_conf.get("USE_TRANSFORMER_ENCODER", False))
        self.trans_dim = int(dic_agent_conf.get("TRANS_DIM", 0))
        self.trans_heads = int(dic_agent_conf.get("TRANS_HEADS", 4))
        self.trans_layers = int(dic_agent_conf.get("TRANS_LAYERS", 2))
        self.trans_ffn_dim = int(dic_agent_conf.get("TRANS_FFN_DIM", 128))
        self.trans_dropout = float(dic_agent_conf.get("TRANS_DROPOUT", 0.1))
        self.trans_use_cos_mask = bool(dic_agent_conf.get("TRANS_USE_COS_MASK", True))
        self.trans_prenorm = bool(dic_agent_conf.get("TRANS_PRENORM", True))
        self.cos_prob_model = None
        
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
                print("[REDQ] enabled, M={}, λ={} (Q_mix = (1-λ)*mean + λ*min_sub)".format(
                    self.redq_m, self.redq_lambda))
        if self.use_competitive_agg:
            print("[CompetitiveAgg] enabled, splitting neighbors into 2 competing groups")
        if self.use_transformer_encoder:
            print(
                "[Transformer] enabled, dim={}, heads={}, layers={}, ffn_dim={}, "
                "dropout={}, use_mask={}, prenorm={}".format(
                    self.trans_dim if self.trans_dim > 0 else self.CNN_layers[0][1],
                    self.trans_heads,
                    self.trans_layers,
                    self.trans_ffn_dim,
                    self.trans_dropout,
                    self.trans_use_cos_mask,
                    self.trans_prenorm,
                )
            )
        if self.cos_enabled:
            print("[CoS] enabled, K={}, include_self={}, beta_diag={}, gamma_sym={}, ent_coef={}".format(
                self.num_neighbors,
                self.cos_include_self,
                self.cos_beta_diag,
                self.cos_gamma_sym,
                self.cos_entropy_coef,
            ))

        if cnt_round == 0:
            # initialization
            self.q_network = self.build_network()
            self._refresh_cos_prob_model()
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.q_network.load_weights(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(intersection_id)),
                    by_name=True)
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

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def _cal_len_feature(self):
        N = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                N += 8
            else:
                N += 12
        return N

    @staticmethod
    def MLP(ins, layers=None):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,dim]
        -outpout: [batch,#agents,dim]
        """
        if layers is None:
            layers = [128, 128]
        for layer_index, layer_size in enumerate(layers):
            if layer_index == 0:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(ins)
            else:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(h)

        return h

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

    def _transformer_encoder_stack(self, x, attn_mask, d_model):
        """
        CoSLight-style Transformer encoder stack on top of agent features.
        x: [B, N, D], attn_mask: [B, N, N]
        """
        h = x
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
        # [batch,agents,neighbors]
        adjacency_index_new = np.sort(adjacency_index, axis=-1)
        lab = to_categorical(adjacency_index_new, num_classes=self.num_agents)
        return lab

    def convert_state_to_input(self, s):
        """
        s: [state1, state2, ..., staten]
        """
        # TODO
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        feats0 = []
        adj = []
        for i in range(self.num_agents):
            adj.append(s[i]["adjacency_matrix"])
            tmp = []
            for feature in used_feature:
                if feature == "cur_phase":
                    if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                        tmp.extend(self.dic_traffic_env_conf['PHASE'][s[i][feature][0]])
                    else:
                        tmp.extend(s[i][feature])
                else:
                    tmp.extend(s[i][feature])

            feats0.append(tmp)
        # [1, agent, dim]
        # feats = np.concatenate([np.array([feat1]), np.array([feat2])], axis=-1)
        feats = np.array([feats0])
        # [1, agent, nei, agent]
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
        q_values = self.q_network(xs)
        if self.cos_enabled and self.cos_prob_model is not None and self.head_debug and count % 50 == 0:
            probs = np.array(self.cos_prob_model.predict(xs, verbose=0))
            ids = self._topk_cos_ids_from_probs(probs)
            print("[CoS] sample ids(inter0):", ids[0, 0].tolist())
        if self.use_multihead:
            # q_values: [1, Agents, N, A]
            q_heads = np.array(q_values[0], dtype=np.float32)  # [Agents, N, A]
            q_mean = np.mean(q_heads, axis=1)  # [Agents, A]
            q_policy = q_mean
            if self.use_ucb_action:
                q_std = np.std(q_heads, axis=1)  # [Agents, A]
                ucb_coef = max(self.ucb_min, self.ucb_lambda * pow(self.ucb_decay, count))
                q_policy = q_mean + ucb_coef * q_std
            if self.head_debug:
                per_head_actions = np.argmax(q_heads, axis=2)  # [Agents, N]
                agree = np.mean([len(set(per_head_actions[a])) == 1
                                 for a in range(per_head_actions.shape[0])])
                print("[MultiHead] head agreement rate: {:.2f}".format(agree))
            if random.random() <= self.dic_agent_conf["EPSILON"]:
                action = np.random.randint(self.num_actions, size=len(q_mean))
            else:
                action = np.argmax(q_policy, axis=1)
        else:
            if random.random() <= self.dic_agent_conf["EPSILON"]:
                action = np.random.randint(self.num_actions, size=len(q_values[0]))
            else:
                action = np.argmax(q_values[0], axis=1)
        return action

    @staticmethod
    def _concat_list(ls):
        tmp = []
        for i in range(len(ls)):
            tmp += ls[i]
        return [tmp]

    def prepare_Xs_Y(self, memory):
        """
        memory: [slice_data, slice_data, ..., slice_data]
        prepare memory for training
        """
        if len(memory) == 0 or len(memory[0]) == 0:
            self.Xs, self.Y = [], []
            return
        slice_size = len(memory[0])
        _adjs = []
        _state = [[] for _ in range(self.num_agents)]
        _next_state = [[] for _ in range(self.num_agents)]
        _action = [[] for _ in range(self.num_agents)]
        _reward = [[] for _ in range(self.num_agents)]

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]

        for i in range(slice_size):
            _adj = []
            for j in range(self.num_agents):
                state, action, next_state, reward, _, _, _ = memory[j][i]
                _action[j].append(action)
                _reward[j].append(reward)
                _adj.append(state["adjacency_matrix"])
                _state[j].append(self._concat_list([state[used_feature[i]] for i in range(len(used_feature))]))
                _next_state[j].append(self._concat_list([next_state[used_feature[i]] for i in range(len(used_feature))]))
            _adjs.append(_adj)
        # [batch, agent, nei, agent]
        _adjs2 = self.adjacency_index2matrix(np.array(_adjs))

        # [batch, 1, dim] -> [batch, agent, dim]
        _state2 = np.concatenate([np.array(ss) for ss in _state], axis=1)
        _next_state2 = np.concatenate([np.array(ss) for ss in _next_state], axis=1)
        target = self.q_network([_state2, _adjs2])
        next_state_qvalues = self.q_network_bar([_next_state2, _adjs2])

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

        self.Xs = [_state2, _adjs2]
        self.Y = final_target

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

    def build_network(self, MLP_layers=[32, 32]):
        CNN_layers = self.CNN_layers
        CNN_heads = [5] * len(CNN_layers)
        In = list()
        # In: [batch,agent,dim]
        # In: [batch,agent,neighbors,agents]
        In.append(Input(shape=(self.num_agents, self.len_feature), name="feature"))
        In.append(Input(shape=(self.num_agents, self.num_neighbors, self.num_agents), name="adjacency_matrix"))

        feature = self.MLP(In[0], MLP_layers)
        att_adj = In[1]
        cos_probs = None
        if self.cos_enabled:
            cos_logits = Dense(
                self.num_agents,
                kernel_initializer='random_normal',
                name='cos_logits'
            )(feature)
            cos_probs = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name="cos_probs")(cos_logits)
            att_adj = CoSDynamicAdjacency(
                num_agents=self.num_agents,
                total_k=self.num_neighbors,
                include_self=self.cos_include_self,
                name="cos_dynamic_adj"
            )(cos_logits)
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
        # action prediction layer
        if self.use_multihead:
            # N independent heads: each [B, Agents, A], stacked → [B, Agents, N, A]
            heads = [Dense(self.num_actions, kernel_initializer='random_normal',
                           name='q_head_{}'.format(k))(h) for k in range(self.head_n)]
            out = StackHeads(name='stack_heads')(heads)
            print("[MultiHead] output shape: [B, {}, {}, {}]".format(
                self.num_agents, self.head_n, self.num_actions))
        else:
            # [batch,agent,32]->[batch,agent,action]
            out = Dense(self.num_actions, kernel_initializer='random_normal', name='action_layer')(h)
        # out:[batch,agent,action] or [batch,agent,N,action]
        model = Model(inputs=In, outputs=out)

        if self.cos_enabled and cos_probs is not None:
            diag = tf.linalg.diag_part(cos_probs)
            diag_loss = -tf.reduce_mean(diag)
            sym_loss = tf.reduce_mean(tf.square(cos_probs - tf.transpose(cos_probs, perm=[0, 2, 1])))
            entropy = -tf.reduce_mean(tf.reduce_sum(cos_probs * tf.math.log(cos_probs + 1e-8), axis=-1))
            if self.cos_beta_diag > 0:
                model.add_loss(self.cos_beta_diag * diag_loss)
            if self.cos_gamma_sym > 0:
                model.add_loss(self.cos_gamma_sym * sym_loss)
            if self.cos_entropy_coef > 0:
                model.add_loss(-self.cos_entropy_coef * entropy)
            model.add_metric(diag_loss, name="cos_diag_loss", aggregation="mean")
            model.add_metric(sym_loss, name="cos_sym_loss", aggregation="mean")
            model.add_metric(entropy, name="cos_entropy", aggregation="mean")

        model.compile(optimizer=Adam(lr=self.dic_agent_conf.get("LEARNING_RATE", 0.0005)),
                      loss=self.dic_agent_conf["LOSS_FUNCTION"])
        model.summary()
        return model

    def train_network(self):
        if not hasattr(self, "Y") or self.Y is None or len(self.Y) == 0:
            return
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')
        self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs, shuffle=False,
                           verbose=2, validation_split=0.3, callbacks=[early_stopping])

    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        custom_objs = {
            "RepeatVector3D": RepeatVector3D,
            "StackHeads": StackHeads,
            "CoSDynamicAdjacency": CoSDynamicAdjacency,
            "MatMulLayer": MatMulLayer,
            "PermuteDimensionsLayer": PermuteDimensionsLayer,
            "SoftmaxMatMulLayer": SoftmaxMatMulLayer,
            "MeanMatMulLayer": MeanMatMulLayer,
            "SliceLayer": SliceLayer,
            "ZerosLayer": ZerosLayer
        }
        network = model_from_json(network_structure,
                                  custom_objects=custom_objs)
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])

        return network

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        custom_objs = {
            "RepeatVector3D": RepeatVector3D,
            "StackHeads": StackHeads,
            "CoSDynamicAdjacency": CoSDynamicAdjacency,
            "MatMulLayer": MatMulLayer,
            "PermuteDimensionsLayer": PermuteDimensionsLayer,
            "SoftmaxMatMulLayer": SoftmaxMatMulLayer,
            "MeanMatMulLayer": MeanMatMulLayer,
            "SliceLayer": SliceLayer,
            "ZerosLayer": ZerosLayer
        }
        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects=custom_objs)
        self._refresh_cos_prob_model()
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        custom_objs = {
            "RepeatVector3D": RepeatVector3D,
            "StackHeads": StackHeads,
            "CoSDynamicAdjacency": CoSDynamicAdjacency,
            "MatMulLayer": MatMulLayer,
            "PermuteDimensionsLayer": PermuteDimensionsLayer,
            "SoftmaxMatMulLayer": SoftmaxMatMulLayer,
            "MeanMatMulLayer": MeanMatMulLayer,
            "SliceLayer": SliceLayer,
            "ZerosLayer": ZerosLayer
        }
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects=custom_objs)
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))


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
