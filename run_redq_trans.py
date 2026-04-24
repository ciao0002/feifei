#!/usr/bin/env python3
"""
Review-package runner for the current REDQ-MLP + static collaborator method.

This copy keeps the proven training path intact while isolating it from the
historical experiment entrypoints in the main repository.
"""

import argparse
import os
import time


DATASET_PRESETS = {
    "arterial1x6": {
        "template": "Arterial",
        "road_net": "1_6",
        "traffic_file": "anon_1_6_300_0.3_synthetic.json",
        "exp_prefix": "arterial_1x6",
    },
    "arterial1x6_delay_small": {
        "template": "ArterialDelay",
        "road_net": "1_6_small",
        "traffic_file": "anon_1_6_300_0.3_synthetic.json",
        "exp_prefix": "arterial_1x6_small",
    },
    "arterial1x6_delay_large": {
        "template": "ArterialDelay",
        "road_net": "1_6_large",
        "traffic_file": "anon_1_6_300_0.3_synthetic.json",
        "exp_prefix": "arterial_1x6_large",
    },
    "hz1x1": {
        "template": "Hangzhou",
        "road_net": "1_1",
        "traffic_file": "hangzhou_1x1_kn-hz_18041607_1h.json",
        "exp_prefix": "hz_1x1",
    },
    "jnreal": {
        "template": "Jinan",
        "road_net": "3_4",
        "traffic_file": "anon_3_4_jinan_real.json",
        "exp_prefix": "jn_real",
    },
    "jnreal_delay_small": {
        "template": "JinanDelay",
        "road_net": "3_4_small",
        "traffic_file": "anon_3_4_jinan_real.json",
        "exp_prefix": "jn_real_small",
    },
    "jnreal_delay_large": {
        "template": "JinanDelay",
        "road_net": "3_4_large",
        "traffic_file": "anon_3_4_jinan_real.json",
        "exp_prefix": "jn_real_large",
    },
    "jn2000": {
        "template": "Jinan",
        "road_net": "3_4",
        "traffic_file": "anon_3_4_jinan_real_2000.json",
        "exp_prefix": "jn_2000",
    },
    "jn2500": {
        "template": "Jinan",
        "road_net": "3_4",
        "traffic_file": "anon_3_4_jinan_real_2500.json",
        "exp_prefix": "jn_2500",
    },
    "hz5816": {
        "template": "Hangzhou",
        "road_net": "4_4",
        "traffic_file": "anon_4_4_hangzhou_real_5816.json",
        "exp_prefix": "hz_5816",
    },
    "hzreal": {
        "template": "Hangzhou",
        "road_net": "4_4",
        "traffic_file": "anon_4_4_hangzhou_real.json",
        "exp_prefix": "hz_real",
    },
    "hzreal_delay_small": {
        "template": "HangzhouDelay",
        "road_net": "4_4_small",
        "traffic_file": "anon_4_4_hangzhou_real.json",
        "exp_prefix": "hz_real_small",
    },
    "hzreal_delay_large": {
        "template": "HangzhouDelay",
        "road_net": "4_4_large",
        "traffic_file": "anon_4_4_hangzhou_real.json",
        "exp_prefix": "hz_real_large",
    },
    "newyork": {
        "template": "newyork_28_7",
        "road_net": "28_7",
        "traffic_file": "anon_28_7_newyork_real_double.json",
        "exp_prefix": "newyork_28_7",
    },
    "manhattan16x3": {
        "template": "Manhattan_16_3",
        "road_net": "16_3",
        "traffic_file": "anon_16_3_newyork_real.json",
        "exp_prefix": "ny_16_3",
    },
    "manhattan16x3_delay_small": {
        "template": "ManhattanDelay",
        "road_net": "16_3_small",
        "traffic_file": "anon_16_3_newyork_real.json",
        "exp_prefix": "ny_16_3_small",
    },
    "manhattan16x3_delay_large": {
        "template": "ManhattanDelay",
        "road_net": "16_3_large",
        "traffic_file": "anon_16_3_newyork_real.json",
        "exp_prefix": "ny_16_3_large",
    },
}


def _max_grid_nodes_within_hop(num_row, num_col, max_hop):
    if max_hop < 0:
        return num_row * num_col
    best = 1
    for r in range(num_row):
        for c in range(num_col):
            cnt = 0
            for rr in range(num_row):
                for cc in range(num_col):
                    if abs(rr - r) + abs(cc - c) <= max_hop:
                        cnt += 1
            best = max(best, cnt)
    return best


def _build_reward_info(reward_type, new_plan, args=None):
    """Build DIC_REWARD_INFO based on reward_type arg or plan-level override."""
    if args is not None:
        custom_reward = {}
        queue_length_weight = args.queue_length_weight
        if queue_length_weight is None:
            queue_length_weight = -0.25
        custom_reward["queue_length"] = float(queue_length_weight)
        if float(args.queue_balance_weight) != 0.0:
            custom_reward["queue_balance"] = float(args.queue_balance_weight)
        if float(args.switch_penalty_weight) != 0.0:
            custom_reward["switch_penalty"] = float(args.switch_penalty_weight)
        if float(args.advanced_pressure_weight) != 0.0:
            custom_reward["advanced_pressure"] = float(args.advanced_pressure_weight)
        if float(args.queue_max_weight) != 0.0:
            custom_reward["queue_max"] = float(args.queue_max_weight)
        if args.soft_fairness_metric != "none" and float(args.soft_fairness_weight) != 0.0:
            custom_reward[str(args.soft_fairness_metric)] = float(args.soft_fairness_weight)
        if any(
            key in custom_reward
            for key in (
                "queue_balance", "switch_penalty", "advanced_pressure",
                "queue_max", "queue_top2_mean", "queue_top3_mean", "queue_pnorm"
            )
        ):
            return custom_reward

    # Plan-level overrides take priority over reward_type arg.
    if new_plan in ("N_pressure",):
        return {"pressure": -1.0}
    if new_plan in ("M_ifdg", "O_ifdg_nf40"):
        return {"ifdg": -1.0}
    if new_plan == "P_hybrid":
        return {"queue_length": -0.25, "ifdg": -0.05}
    if new_plan == "AC_queue_switch":
        return {"queue_length": -0.25, "switch_penalty": -0.10}
    if new_plan == "AD_queue_switch_fair":
        return {"queue_length": -0.25, "switch_penalty": -0.10, "queue_max": -0.10}
    if new_plan == "AE_queue_switch_balance":
        return {"queue_length": -0.25, "switch_penalty": -0.10, "queue_balance": -0.10}
    if new_plan == "AI_qsb_adv":
        return {"queue_length": -0.25, "switch_penalty": -0.20, "queue_balance": -0.05, "advanced_pressure": -0.05}
    if new_plan == "Y_queuefair":
        return {"queue_length": -0.25, "queue_max": -0.10}
    if new_plan == "Z_queuebalance":
        return {"queue_length": -0.25, "queue_balance": -0.10}
    if new_plan == "R_staged_ifdg":
        return {"queue_length": -0.25, "ifdg": -0.05}
    # Explicit reward_type arg.
    if reward_type == "ifdg":
        return {"ifdg": -1.0}
    if reward_type == "pressure":
        return {"pressure": -1.0}
    if reward_type == "regional_queue":
        return {"regional_queue": -0.25}
    return {"queue_length": -0.25}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default="jnreal", choices=sorted(DATASET_PRESETS.keys()))
    parser.add_argument("-traffic_file", type=str, default=None)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-num_rounds", type=int, default=60)
    parser.add_argument("-run_counts", type=int, default=3600)
    parser.add_argument("-num_generators", type=int, default=1)
    parser.add_argument("-generator_cuda_visible_devices", type=str, default=None,
                        help="Optional CUDA_VISIBLE_DEVICES for generator workers only.")
    parser.add_argument("-memo_prefix", type=str, default="trueredq_trans_cuda")
    parser.add_argument("-model_name", type=str, default="REDQ",
                        choices=["REDQ"],
                        help="Registered model name in utils.config.DIC_AGENTS.")
    parser.add_argument(
        "-ablation_mode",
        type=str,
        default="mlp_only",
        choices=["full", "no_trans", "mlp_only", "std_trans", "no_redq", "mlp_no_redq"],
        help=(
            "Ablation preset: "
            "full=REDQ+graph-masked Transformer, "
            "no_trans=REDQ+GAT aggregation, "
            "mlp_only=REDQ+MLP only (no Transformer/GAT), "
            "std_trans=REDQ+fully-visible Transformer, "
            "no_redq=DQN-style update+graph-masked Transformer; "
            "mlp_no_redq=DQN-style update+MLP-only."
        ),
    )
    parser.add_argument(
        "-new_plan",
        type=str,
        default="base",
        choices=["base", "A_utd", "B_soft", "C_epochs", "D_replay", "E_combo",
                 "F_dueling", "G_double", "H_2step", "I_duel_double", "J_all",
                 "K_stable", "L_floor", "M_ifdg", "N_pressure", "O_ifdg_nf40",
                 "P_hybrid", "R_staged_ifdg",
                 "AC_queue_switch", "AD_queue_switch_fair", "AE_queue_switch_balance",
                 "Y_queuefair", "Z_queuebalance", "AI_qsb_adv", "P_aplight"],
        help=(
            "Improvement plan selector. "
            "base=no change; "
            "A_utd=true UTD=4; B_soft=soft target τ=0.01; C_epochs=3 epochs+lr=0.0005; "
            "D_replay=30k buffer+5k sample; E_combo=A+B+C+D+min_ε=0.05; "
            "F_dueling=Dueling DQN; G_double=Double DQN; H_2step=2-step return; "
            "I_duel_double=F+G; J_all=F+G+H; K_stable=H+B; L_floor=I+B; "
            "M_ifdg=L_floor+IFDG reward; N_pressure=L_floor+pressure reward; "
            "O_ifdg_nf40=M+NF=40; P_hybrid=L_floor+queue+small IFDG; "
            "AC_queue_switch=L_floor+queue+switch penalty only; "
            "AD_queue_switch_fair=L_floor+queue+switch penalty+worst-lane queue; "
            "AE_queue_switch_balance=L_floor+queue+switch penalty+queue balance; "
            "Y_queuefair=L_floor+queue+worst-lane queue penalty; "
            "Z_queuebalance=L_floor+queue+queue-balance penalty; "
            "R_staged_ifdg=L_floor+queue then gradual IFDG blend; "
            "AI_qsb_adv=L_floor+queue+switch+balance+advanced pressure; "
            "P_aplight=L_floor+adaptive phase pressure head."
        ),
    )
    parser.add_argument("-max_memory_len", type=int, default=None,
                        help="MAX_MEMORY_LEN override (default from config: 12000).")

    parser.add_argument("-epochs", type=int, default=1)
    parser.add_argument("-batch_size", type=int, default=20)
    parser.add_argument("-sample_size", type=int, default=3000)
    parser.add_argument("-min_epsilon", type=float, default=0.2)
    parser.add_argument("-gamma", type=float, default=0.8,
                        help="Discount factor γ (default 0.8). Suggested: 0.95 for richer credit assignment.")
    parser.add_argument("-normal_factor", type=float, default=20.0,
                        help="Reward normalization η (default 20). Suggested: 10 when γ=0.95.")

    parser.add_argument("-redq_n", type=int, default=4)
    parser.add_argument("-redq_m", type=int, default=2)
    parser.add_argument("-redq_utd", type=int, default=4)
    parser.add_argument("-redq_utd_warmup_rounds", type=int, default=0,
                        help="If >0, use REDQ_UTD_WARMUP_VALUE for the first N rounds, then switch to REDQ_UTD_AFTER_VALUE.")
    parser.add_argument("-redq_utd_warmup_value", type=int, default=None,
                        help="Warm-up UTD used before REDQ_UTD_WARMUP_ROUNDS. Defaults to -redq_utd when omitted.")
    parser.add_argument("-redq_utd_after_value", type=int, default=None,
                        help="UTD used after warm-up rounds. Defaults to -redq_utd when omitted.")
    parser.add_argument("-redq_lambda", type=float, default=1.0)
    parser.add_argument(
        "-deterministic_redq_acting",
        action="store_true",
        help="When using true REDQ ensemble, keep epsilon exploration but remove random subset sampling in acting; use deterministic all-critic Q_mix for action selection.",
    )
    parser.add_argument("-redq_paper_utd", action="store_true",
                        help="Use paper-style UTD: resample replay for every UTD update, including the first one.")
    parser.add_argument("-redq_soft_target", action="store_true",
                        help="Use Polyak soft target updates (paper-style) instead of lagged hard target copy.")
    parser.add_argument("-redq_tau", type=float, default=0.005,
                        help="Polyak coefficient tau for soft target update: target=(1-tau)*target+tau*online.")
    parser.add_argument("-droq_mode", action="store_true",
                        help="Enable DroQ-style critic regularization defaults (LayerNorm + Dropout).")
    parser.add_argument("-critic_use_layer_norm", action="store_true",
                        help="Apply LayerNorm after each critic hidden dense layer.")
    parser.add_argument("-critic_dropout_rate", type=float, default=0.0,
                        help="Dropout rate applied after each critic hidden activation. 0 disables dropout.")
    parser.add_argument("-critic_hidden_dim", type=int, default=32,
                        help="Hidden width used by the critic MLP trunk.")
    parser.add_argument("-critic_num_layers", type=int, default=2,
                        help="Number of hidden layers used by the critic MLP trunk.")
    parser.add_argument("-crossq_safe_mode", action="store_true",
                        help="Enable CrossQ-safe critic update inside the existing true REDQ path.")
    parser.add_argument("-crossq_bn_mode", type=str, default="brn", choices=["bn", "brn"],
                        help="Normalization type used by CrossQ-safe critic trunk.")
    parser.add_argument("-crossq_use_batch_norm", action="store_true",
                        help="Enable BN/BRN inside critic hidden trunk for CrossQ-safe mode.")
    parser.add_argument("-crossq_batch_norm_momentum", type=float, default=0.99,
                        help="Momentum for CrossQ-safe BN/BRN moving statistics.")
    parser.add_argument("-crossq_brn_warmup_steps", type=int, default=100000,
                        help="Warm-up steps before BRN renorm correction is enabled.")
    parser.add_argument("-crossq_use_live_bnstats_for_target", action="store_true",
                        help="Use live online BN stats for target path. Default off for safe mode.")
    parser.add_argument("-crossq_joint_forward", action="store_true",
                        help="Use current+next joint forward pass during critic update.")
    parser.add_argument("-crossq_custom_train_step", action="store_true",
                        help="Use custom GradientTape critic update instead of fit()/predict() in true REDQ mode.")
    parser.add_argument("-crossq_keep_target_net", action="store_true",
                        help="Keep target network enabled in CrossQ-safe mode.")
    parser.add_argument("-use_per", action="store_true",
                        help="Enable prioritized replay (PER) for REDQ training.")
    parser.add_argument("-per_alpha", type=float, default=0.6,
                        help="PER alpha in proportional prioritization.")
    parser.add_argument("-per_eps", type=float, default=1e-3,
                        help="Small epsilon added to TD-error before exponent in PER.")
    parser.add_argument("-per_uniform_mix", type=float, default=0.1,
                        help="Mixture ratio with uniform sampling to avoid over-focus in PER.")
    parser.add_argument("-per_pool_mult", type=int, default=4,
                        help="Candidate pool multiplier for PER. Updater samples SAMPLE_SIZE*mult, then PER resamples.")
    parser.add_argument("-per_beta", type=float, default=0.4,
                        help="PER beta for importance sampling weights. Default 0.4; use 0 to disable IS weighting.")
    parser.add_argument("-use_noisy_net", action="store_true",
                        help="Enable NoisyNet parameter noise in the MLP trunk and Q heads.")
    parser.add_argument("-noisy_sigma_init", type=float, default=0.017,
                        help="Initial sigma for NoisyNet layers.")
    parser.add_argument("-use_ucb_action", action="store_true",
                        help="Enable UCB action selection using critic ensemble mean/std.")
    parser.add_argument("-ucb_lambda", type=float, default=0.2,
                        help="UCB coefficient multiplying critic std.")
    parser.add_argument("-ucb_decay", type=float, default=1.0,
                        help="UCB decay factor over rounds.")
    parser.add_argument("-ucb_min", type=float, default=0.0,
                        help="Lower bound for UCB coefficient.")
    parser.add_argument("-action_gaussian_std", type=float, default=0.0,
                        help="Std of Gaussian noise added to greedy action logits before argmax.")
    parser.add_argument("-action_gaussian_clip", type=float, default=0.0,
                        help="Optional absolute clip for Gaussian action-logit noise. 0 disables clipping.")
    parser.add_argument("-trans_dim", type=int, default=32)
    parser.add_argument("-trans_heads", type=int, default=4)
    parser.add_argument("-trans_layers", type=int, default=2)
    parser.add_argument("-trans_ffn_dim", type=int, default=128)
    parser.add_argument("-trans_dropout", type=float, default=0.1)
    parser.add_argument("-disable_trans_cos_mask", action="store_true")
    parser.add_argument("-disable_trans_prenorm", action="store_true")
    parser.add_argument("-use_block_attn_res", action="store_true",
                        help="Enable lightweight Block AttnRes-style cross-block residual aggregation in Transformer mode.")
    parser.add_argument("-enable_cos", action="store_true",
                        help="Enable CoS top-k collaborator selection module.")
    parser.add_argument("-use_feature_group_gate", action="store_true",
                        help="Enable feature-group gate encoder before critic heads.")
    parser.add_argument("-use_feature_group_concat", action="store_true",
                        help="Enable grouped-feature split+concat encoder control instead of plain MLP.")
    parser.add_argument("-feature_group_hidden_dim", type=int, default=16,
                        help="Hidden dim for each feature-group encoder branch.")
    parser.add_argument("-use_auxiliary_head", action="store_true",
                        help="Enable an auxiliary prediction head on top of the shared critic representation.")
    parser.add_argument("-auxiliary_task", type=str, default="none",
                        choices=["none", "reward", "next_pressure", "latent_transition"],
                        help="Auxiliary task type when auxiliary head is enabled.")
    parser.add_argument("-auxiliary_weight", type=float, default=0.0,
                        help="Loss weight alpha for the auxiliary head.")
    parser.add_argument("-auxiliary_ema_tau", type=float, default=0.995,
                        help="EMA coefficient for latent-transition target encoder.")
    parser.add_argument("-cos_total_k", type=int, default=5,
                        help="Total collaborators K for CoS (including self when enabled).")
    parser.add_argument("-cos_adj_mode", type=str, default="tiled_sparse",
                        choices=["tiled_sparse", "topk_slots"],
                        help="Dynamic collaborator adjacency mode.")
    parser.add_argument("-cos_slot_min_prob", type=float, default=0.0,
                        help="Minimum collaborator probability threshold used by topk_slots to softly suppress weak slots.")
    parser.add_argument("-cos_use_input_candidate_mask", action="store_true",
                        help="Restrict dynamic collaborator scoring to the candidate set carried by adjacency_matrix.")
    parser.add_argument("-neighbor_select_enabled", action="store_true",
                        help="Enable relation-based dynamic neighbor selection and aggregation before critic/Q head.")
    parser.add_argument("-neighbor_candidate_hop", type=int, default=2,
                        help="Candidate hop budget used for relation-based neighbor selection.")
    parser.add_argument("-neighbor_topk", type=int, default=5,
                        help="Maximum number of selected collaborators in the relation-based selector.")
    parser.add_argument("-neighbor_gate_type", type=str, default="soft", choices=["hard", "soft"],
                        help="Gate type for selected top-k collaborators.")
    parser.add_argument("-neighbor_gate_threshold", type=float, default=0.1,
                        help="Threshold used by hard/soft gate after top-k selection.")
    parser.add_argument("-neighbor_gate_temp", type=float, default=0.05,
                        help="Temperature for soft gate.")
    parser.add_argument("-use_topo_feature", action="store_true",
                        help="Use topology relation features in the relation-based selector.")
    parser.add_argument("-use_delay_feature", action="store_true",
                        help="Use delay/distance relation features in the relation-based selector.")
    parser.add_argument("-use_same_corridor_feature", action="store_true",
                        help="Use same-corridor relation feature when topology features are enabled.")
    parser.add_argument("-delay_use_distance_only", action="store_true",
                        help="Use distance only as delay proxy instead of distance+estimated delay.")
    parser.add_argument("-relation_hidden_dim", type=int, default=32,
                        help="Hidden dim for relation scoring MLP.")
    parser.add_argument("-neighbor_state_rel_mode", type=str, default="diff_only",
                        choices=["none", "diff_only", "pair", "full"],
                        help="State relation feature mode used in neighbor selector.")
    parser.add_argument("-use_mlp_neighbor_agg", action="store_true",
                        help="Enable residual mean neighbor aggregation in mlp_only mode.")
    parser.add_argument("-use_neighbor_h_mean_concat", action="store_true",
                        help="Use StaticDelay external-neighbor hidden mean pooling: z_i=concat(h_i, mean_j h_j).")
    parser.add_argument("-use_delay_msg_mean", action="store_true",
                        help="Use StaticDelay neighbor message mean pooling: u_ij=MLP([h_j,tau_ij]).")
    parser.add_argument("-use_delay_rel_msg_mean", action="store_true",
                        help="Use StaticDelay relation-aware neighbor message mean pooling: u_ij=MLP([h_j,tau_ij,delta_p,delta_r]).")
    parser.add_argument("-delay_msg_hidden_dim", type=int, default=32,
                        help="Hidden dimension of the lightweight neighbor message MLP.")
    parser.add_argument("-delay_msg_tau_norm_mode", type=str, default="min_action_time",
                        choices=["min_action_time"],
                        help="Tau normalization mode for delay message aggregation.")
    parser.add_argument("-delay_msg_delta_reduce", type=str, default="mean",
                        choices=["mean"],
                        help="How to compress pressure/running-part deltas for delay relation message aggregation.")
    parser.add_argument("-critic_activation", type=str, default="relu",
                        choices=["relu", "sigmoid", "linear"],
                        help="Hidden activation used by REDQ-MLP backbone and delay-message MLPs.")
    parser.add_argument("-use_dynamic_collab", action="store_true",
                        help="Explicit alias for mlp_only dynamic collaborator selection: enable CoS adjacency and residual neighbor aggregation together.")
    parser.add_argument("-use_dynamic_collab_full", action="store_true",
                        help="Enable a fuller dynamic collaborator selector with local need-gate and pairwise collaborator scoring.")
    parser.add_argument("-dynamic_collab_pair_dim", type=int, default=32,
                        help="Hidden dimension for pairwise collaborator scoring MLP.")
    parser.add_argument("-dynamic_collab_need_bias", type=float, default=2.0,
                        help="Self-collaboration bias added when local collaboration need is low.")
    parser.add_argument("-cos_beta_diag", type=float, default=0.0,
                        help="CoS diagonal regularization weight.")
    parser.add_argument("-cos_gamma_sym", type=float, default=0.0,
                        help="CoS symmetry regularization weight.")
    parser.add_argument("-cos_entropy_coef", type=float, default=0.0,
                        help="CoS entropy regularization coefficient. Positive values encourage sharper collaborator selection.")
    parser.add_argument("-cos_budget_coef", type=float, default=0.0,
                        help="CoS sparsity / budget regularization coefficient.")
    parser.add_argument("-cos_budget_thr", type=float, default=0.0,
                        help="Threshold used by CoS budget regularizer to count active edges.")
    parser.add_argument("-cos_budget_tau", type=float, default=0.05,
                        help="Smoothness temperature for CoS budget regularizer.")
    parser.add_argument("-use_intersection_pos_enc", action="store_true",
                        help="Inject learned intersection positional embedding from static topology vector.")
    parser.add_argument("-intersection_pos_dim", type=int, default=16,
                        help="Hidden dimension for learned intersection positional embedding.")
    parser.add_argument("-max_hop", type=int, default=-1,
                        help="Keep only nodes within this hop distance in adjacency_matrix. -1 disables hop masking.")
    parser.add_argument("-distance_topk_mode", action="store_true",
                        help="Use distance-sorted top-k adjacency instead of hop masking.")
    parser.add_argument("-distance_topk_k", type=int, default=5,
                        help="Top-k size used when distance_topk_mode is enabled.")
    parser.add_argument("-static_delay_candidate_mode", action="store_true",
                        help="Use shortest-path delay-feasible candidate set with fixed padding.")
    parser.add_argument("-static_delay_multiplier", type=float, default=1.0,
                        help="Delay threshold multiplier applied to MIN_ACTION_TIME when static delay candidates are enabled.")
    parser.add_argument("-static_delay_candidate_rmax", type=int, default=8,
                        help="Fixed candidate-slot count (including self) for static delay candidate mode.")
    parser.add_argument("-static_delay_min_external", type=int, default=0,
                        help="Minimum number of external neighbors to keep by nearest-distance fallback in static delay candidate mode.")
    parser.add_argument("-mask_farthest_count", type=int, default=0,
                        help="Mask this many farthest nodes inside the distance-sorted top-k set.")
    parser.add_argument("-reward_type", type=str, default="queue",
                        choices=["queue", "regional_queue", "pressure", "ifdg"],
                        help="Reward signal: queue=incoming queue_length, regional_queue=incoming+outgoing queue, pressure=intersection pressure, ifdg=IFDG unbiased ATT proxy.")
    parser.add_argument(
        "-feature_set",
        type=str,
        default="baseline",
        choices=["baseline", "baseline_hist2", "coslight", "ats_pro_v2", "ats_las"],
        help=(
            "State feature preset. "
            "baseline=[cur_phase,efficient_pressure,running_part], "
            "baseline_hist2=baseline + previous-step copies of phase/pressure/running, "
            "coslight=richer local state for gate experiments, "
            "ats_pro_v2=ATS-Pro 7-d movement state, "
            "ats_las=ATS + lane average speed (LAS)."
        ),
    )
    parser.add_argument("-queue_length_weight", type=float, default=None,
                        help="Override queue_length reward weight. Default keeps plan/default value.")
    parser.add_argument("-queue_balance_weight", type=float, default=0.0,
                        help="Additional queue_balance reward weight.")
    parser.add_argument("-switch_penalty_weight", type=float, default=0.0,
                        help="Additional switch_penalty reward weight.")
    parser.add_argument("-advanced_pressure_weight", type=float, default=0.0,
                        help="Additional advanced_pressure reward weight.")
    parser.add_argument("-queue_max_weight", type=float, default=0.0,
                        help="Additional worst-lane queue reward weight.")
    parser.add_argument("-soft_fairness_metric", type=str, default="none",
                        choices=["none", "queue_top2_mean", "queue_top3_mean", "queue_pnorm"],
                        help="Optional soft fairness metric added into reward.")
    parser.add_argument("-soft_fairness_weight", type=float, default=0.0,
                        help="Reward weight for soft fairness metric.")
    parser.add_argument("-add_phase_elapsed", action="store_true")
    parser.add_argument("-add_delta_pressure", action="store_true")
    parser.add_argument("-phase_elapsed_norm_base", type=float, default=15.0)
    parser.add_argument("-add_downstream_congestion", action="store_true")
    parser.add_argument(
        "-ema_mode",
        type=str,
        default="none",
        choices=["none", "pressure", "running", "both", "raw_plus_both", "raw_plus_pressure"],
        help="Inject EMA-smoothed dynamic features before MLP. raw_plus_both keeps raw and EMA copies together.",
    )
    parser.add_argument(
        "-ema_alpha",
        type=float,
        default=0.4,
        help="EMA smoothing coefficient alpha for dynamic features.",
    )
    parser.add_argument(
        "-use_kalman_pressure",
        action="store_true",
        help="Replace efficient pressure with a Kalman-filtered version in baseline feature_set.",
    )
    parser.add_argument(
        "-use_raw_plus_kalman_pressure",
        action="store_true",
        help="Keep both raw and Kalman-filtered pressure as dual inputs in baseline feature_set.",
    )
    parser.add_argument(
        "-kalman_pressure_q",
        type=float,
        default=0.05,
        help="Process noise Q for pressure Kalman filter.",
    )
    parser.add_argument(
        "-kalman_pressure_r",
        type=float,
        default=1.0,
        help="Observation noise R for pressure Kalman filter.",
    )
    parser.add_argument("-disable_critic_bootstrap_sample", action="store_true",
                        help="Disable per-critic bootstrap resampling and train each critic on the same minibatch.")

    parser.add_argument("-cuda_visible_devices", type=str, default="0")
    parser.add_argument("-no_gpu_memory_growth", action="store_true")
    parser.add_argument("-enable_xla", action="store_true")
    parser.add_argument("-mixed_precision", action="store_true")
    parser.add_argument("-require_gpu", action="store_true", help="Fail fast if no visible GPU.")

    parser.add_argument("-validate_only", action="store_true", help="Only check runtime setup and print config.")
    parser.add_argument("-resume_run_dir", type=str, default=None,
                        help="Existing records run dir to resume from (e.g. records/.../<exp_id>).")
    parser.add_argument("-resume_model_dir", type=str, default=None,
                        help="Existing model run dir to resume from. If omitted, auto-derive from resume_run_dir.")
    return parser.parse_args()


def configure_tf_runtime(args):
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if not args.no_gpu_memory_growth:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if args.enable_xla:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not args.no_gpu_memory_growth:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    if args.mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    if args.require_gpu and not gpus:
        raise RuntimeError("No GPU detected by TensorFlow. Check CUDA/cuDNN runtime and CUDA_VISIBLE_DEVICES.")
    return tf.__version__, gpus


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if args.num_generators > 1:
        # Avoid forking a process with an already-initialized TF CUDA runtime.
        from multiprocessing import set_start_method
        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    tf_version, gpus = configure_tf_runtime(args)
    if args.use_feature_group_gate and args.use_feature_group_concat:
        raise ValueError("Choose only one of use_feature_group_gate or use_feature_group_concat.")
    if args.use_dynamic_collab:
        args.enable_cos = True
        args.use_mlp_neighbor_agg = True
    if args.use_dynamic_collab_full:
        args.enable_cos = True
        args.use_mlp_neighbor_agg = True

    preset = DATASET_PRESETS[args.dataset]
    road_net = preset["road_net"]
    traffic_file = args.traffic_file or preset["traffic_file"]
    template = preset["template"]
    num_row = int(road_net.split("_")[0])
    num_col = int(road_net.split("_")[1])

    from utils import config
    from utils.utils import merge
    from utils.pipeline import Pipeline

    plan_suffix = "" if args.new_plan == "base" else "_{}".format(args.new_plan)
    memo = "{}_{}_s{}_n{}m{}u{}{}".format(
        args.memo_prefix, args.dataset, args.seed, args.redq_n, args.redq_m, args.redq_utd, plan_suffix
    )
    exp_id = "trueredq_trans_cuda_{}_{}".format(
        preset["exp_prefix"], time.strftime("%m%d_%H%M%S", time.localtime(time.time()))
    )

    resume_run_dir = args.resume_run_dir
    resume_model_dir = args.resume_model_dir
    if resume_run_dir:
        if resume_model_dir is None:
            norm = os.path.normpath(resume_run_dir)
            if norm.startswith("records" + os.sep):
                resume_model_dir = os.path.join("model", norm.split(os.sep, 1)[1])
            elif "/records/" in norm:
                resume_model_dir = norm.replace("/records/", "/model/", 1)
            else:
                raise ValueError("Cannot infer resume_model_dir from resume_run_dir, please provide -resume_model_dir")

    if args.feature_set == "coslight":
        list_state_feature = [
            "cur_phase",
            "phase_elapsed",
            "lane_num_vehicle",
            "lane_num_vehicle_close",
            "lane_num_waiting_vehicle_in",
            "traffic_movement_pressure_num",
            "lane_enter_running_part",
        ]
    elif args.feature_set == "baseline_hist2":
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "lane_enter_running_part",
            "cur_phase_previous_step",
            "traffic_movement_pressure_queue_efficient_previous_step",
            "lane_enter_running_part_previous_step",
        ]
    elif args.feature_set == "ats_pro_v2":
        list_state_feature = [
            "cur_phase",
            "ats_effective_pressure",
            "ats_effective_running_demand",
            "ats_far_approaching_demand",
            "queue_growth_rate_movement",
            "queue_decay_rate_movement",
            "downstream_saturation_movement",
            "weighted_accumulated_wait_movement",
        ]
    elif args.feature_set == "ats_las":
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "lane_enter_running_part",
            "ats_lane_average_speed",
        ]
    else:
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "lane_enter_running_part",
        ]
    if args.use_kalman_pressure and args.use_raw_plus_kalman_pressure:
        raise ValueError("Choose only one of use_kalman_pressure or use_raw_plus_kalman_pressure.")
    if args.use_kalman_pressure:
        if args.feature_set != "baseline":
            raise ValueError("Kalman pressure currently only supports feature_set=baseline.")
        if args.ema_mode != "none":
            raise ValueError("Kalman pressure cannot be combined with ema_mode.")
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient_kalman",
            "lane_enter_running_part",
        ]
    if args.use_raw_plus_kalman_pressure:
        if args.feature_set != "baseline":
            raise ValueError("Raw+Kalman pressure currently only supports feature_set=baseline.")
        if args.ema_mode != "none":
            raise ValueError("Raw+Kalman pressure cannot be combined with ema_mode.")
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "traffic_movement_pressure_queue_efficient_kalman",
            "lane_enter_running_part",
        ]
    if args.ema_mode != "none" and args.feature_set != "baseline":
        raise ValueError("EMA feature smoothing currently only supports feature_set=baseline.")
    if args.ema_mode == "pressure":
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient_ema",
            "lane_enter_running_part",
        ]
    elif args.ema_mode == "running":
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "lane_enter_running_part_ema",
        ]
    elif args.ema_mode == "both":
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient_ema",
            "lane_enter_running_part_ema",
        ]
    elif args.ema_mode == "raw_plus_both":
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "traffic_movement_pressure_queue_efficient_ema",
            "lane_enter_running_part",
            "lane_enter_running_part_ema",
        ]
    elif args.ema_mode == "raw_plus_pressure":
        list_state_feature = [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "traffic_movement_pressure_queue_efficient_ema",
            "lane_enter_running_part",
        ]
    add_phase_elapsed = bool(args.add_phase_elapsed)
    if args.feature_set == "coslight":
        add_phase_elapsed = False
    if add_phase_elapsed:
        list_state_feature.append("phase_elapsed")
    if args.add_delta_pressure:
        list_state_feature.append("delta_pressure")
    if args.add_downstream_congestion:
        list_state_feature.append("downstream_congestion")
    if args.use_intersection_pos_enc and "intersection_topology_vector" not in list_state_feature:
        list_state_feature.append("intersection_topology_vector")
    if args.use_dynamic_collab_full and "intersection_topology_vector" not in list_state_feature:
        list_state_feature.append("intersection_topology_vector")
    if args.neighbor_select_enabled and (args.use_topo_feature or args.use_delay_feature or args.use_same_corridor_feature):
        if "intersection_topology_vector" not in list_state_feature:
            list_state_feature.append("intersection_topology_vector")
    list_state_feature.append("adjacency_matrix")

    critic_dropout_rate = float(args.critic_dropout_rate)
    critic_use_layer_norm = bool(args.critic_use_layer_norm)
    if args.droq_mode:
        critic_use_layer_norm = True
        if critic_dropout_rate <= 0:
            critic_dropout_rate = 0.05
    crossq_use_batch_norm = bool(args.crossq_use_batch_norm)
    crossq_joint_forward = bool(args.crossq_joint_forward)
    crossq_custom_train_step = bool(args.crossq_custom_train_step)
    crossq_keep_target_net = bool(args.crossq_keep_target_net)
    if args.crossq_safe_mode:
        crossq_use_batch_norm = True
        crossq_joint_forward = True
        crossq_custom_train_step = True
        crossq_keep_target_net = True
        dic_soft = True
    else:
        dic_soft = bool(args.redq_soft_target)

    dic_agent_conf_extra = {
        "CNN_layers": [[32, 32]],
        "USE_MULTIHEAD_Q": False,
        "HEAD_N": int(args.redq_n),
        "HEAD_AGG": "mean",
        "HEAD_DEBUG": False,
        "USE_UCB_ACTION": False,
        "USE_HEAD_BOOTSTRAP": False,
        "USE_CRITIC_BOOTSTRAP_SAMPLE": not bool(args.disable_critic_bootstrap_sample),
        "GAMMA": float(args.gamma),
        "NORMAL_FACTOR": float(args.normal_factor),
        "MIN_EPSILON": float(args.min_epsilon),
        "EPOCHS": int(args.epochs),
        "BATCH_SIZE": int(args.batch_size),
        "SAMPLE_SIZE": int(args.sample_size),
        "USE_REDQ": True,
        "TRUE_REDQ_MODE": True,
        "REDQ_N": int(args.redq_n),
        "REDQ_M": int(args.redq_m),
        "REDQ_UTD": int(args.redq_utd),
        "REDQ_UTD_WARMUP_ROUNDS": int(args.redq_utd_warmup_rounds),
        "REDQ_UTD_WARMUP_VALUE": int(args.redq_utd_warmup_value) if args.redq_utd_warmup_value is not None else int(args.redq_utd),
        "REDQ_UTD_AFTER_VALUE": int(args.redq_utd_after_value) if args.redq_utd_after_value is not None else int(args.redq_utd),
        "REDQ_LAMBDA": float(args.redq_lambda),
        "DETERMINISTIC_REDQ_ACTING": bool(args.deterministic_redq_acting),
        "REDQ_PAPER_UTD": bool(args.redq_paper_utd),
        "REDQ_SOFT_TARGET_UPDATE": bool(dic_soft),
        "REDQ_TAU": float(args.redq_tau),
        "DROQ_MODE": bool(args.droq_mode),
        "CRITIC_USE_LAYER_NORM": bool(critic_use_layer_norm),
        "CRITIC_DROPOUT_RATE": float(critic_dropout_rate),
        "CRITIC_HIDDEN_DIM": int(args.critic_hidden_dim),
        "CRITIC_NUM_LAYERS": int(args.critic_num_layers),
        "CROSSQ_SAFE_MODE": bool(args.crossq_safe_mode),
        "CROSSQ_BN_MODE": str(args.crossq_bn_mode),
        "CROSSQ_USE_BATCH_NORM": bool(crossq_use_batch_norm),
        "CROSSQ_BATCH_NORM_MOMENTUM": float(args.crossq_batch_norm_momentum),
        "CROSSQ_BRN_WARMUP_STEPS": int(args.crossq_brn_warmup_steps),
        "CROSSQ_USE_LIVE_BNSTATS_FOR_TARGET": bool(args.crossq_use_live_bnstats_for_target),
        "CROSSQ_JOINT_FORWARD": bool(crossq_joint_forward),
        "CROSSQ_CUSTOM_TRAIN_STEP": bool(crossq_custom_train_step),
        "CROSSQ_KEEP_TARGET_NET": bool(crossq_keep_target_net),
        "USE_PER": bool(args.use_per),
        "PER_ALPHA": float(args.per_alpha),
        "PER_EPS": float(args.per_eps),
        "PER_UNIFORM_MIX": float(args.per_uniform_mix),
        "PER_POOL_MULT": int(args.per_pool_mult),
        "PER_BETA": float(args.per_beta),
        "USE_NOISY_NET": bool(args.use_noisy_net),
        "NOISY_SIGMA_INIT": float(args.noisy_sigma_init),
        "USE_UCB_ACTION": bool(args.use_ucb_action),
        "UCB_LAMBDA": float(args.ucb_lambda),
        "UCB_DECAY": float(args.ucb_decay),
        "UCB_MIN": float(args.ucb_min),
        "ACTION_GAUSSIAN_STD": float(args.action_gaussian_std),
        "ACTION_GAUSSIAN_CLIP": float(args.action_gaussian_clip),
        "RELIGHT_ACTION_VOTE": False,
        "COS_ENABLED": bool(args.enable_cos),
        "COS_TOTAL_K": int(args.cos_total_k),
        "COS_ADJ_MODE": str(args.cos_adj_mode),
        "COS_SLOT_MIN_PROB": float(args.cos_slot_min_prob),
        "COS_USE_INPUT_CANDIDATE_MASK": bool(args.cos_use_input_candidate_mask),
        "NEIGHBOR_SELECT_ENABLED": bool(args.neighbor_select_enabled),
        "NEIGHBOR_CANDIDATE_HOP": int(args.neighbor_candidate_hop),
        "NEIGHBOR_TOPK": int(args.neighbor_topk),
        "NEIGHBOR_GATE_TYPE": str(args.neighbor_gate_type),
        "NEIGHBOR_GATE_THRESHOLD": float(args.neighbor_gate_threshold),
        "NEIGHBOR_GATE_TEMP": float(args.neighbor_gate_temp),
        "USE_TOPO_FEATURE": bool(args.use_topo_feature),
        "USE_DELAY_FEATURE": bool(args.use_delay_feature),
        "USE_SAME_CORRIDOR_FEATURE": bool(args.use_same_corridor_feature),
        "DELAY_USE_DISTANCE_ONLY": bool(args.delay_use_distance_only),
        "RELATION_HIDDEN_DIM": int(args.relation_hidden_dim),
        "NEIGHBOR_STATE_REL_MODE": str(args.neighbor_state_rel_mode),
        "USE_DYNAMIC_COLLAB_FULL": bool(args.use_dynamic_collab_full),
        "DYNAMIC_COLLAB_PAIR_DIM": int(args.dynamic_collab_pair_dim),
        "DYNAMIC_COLLAB_NEED_BIAS": float(args.dynamic_collab_need_bias),
        "COS_BETA_DIAG": float(args.cos_beta_diag),
        "COS_GAMMA_SYM": float(args.cos_gamma_sym),
        "COS_ENTROPY_COEF": float(args.cos_entropy_coef),
        "COS_BUDGET_COEF": float(args.cos_budget_coef),
        "COS_BUDGET_THR": float(args.cos_budget_thr),
        "COS_BUDGET_TAU": float(args.cos_budget_tau),
        "USE_MLP_NEIGHBOR_AGG": bool(args.use_mlp_neighbor_agg),
        "USE_NEIGHBOR_H_MEAN_CONCAT": bool(args.use_neighbor_h_mean_concat),
        "USE_DELAY_MSG_MEAN": bool(args.use_delay_msg_mean),
        "USE_DELAY_REL_MSG_MEAN": bool(args.use_delay_rel_msg_mean),
        "DELAY_MSG_HIDDEN_DIM": int(args.delay_msg_hidden_dim),
        "DELAY_MSG_TAU_NORM_MODE": str(args.delay_msg_tau_norm_mode),
        "DELAY_MSG_DELTA_REDUCE": str(args.delay_msg_delta_reduce),
        "CRITIC_ACTIVATION": str(args.critic_activation),
        "USE_INTERSECTION_POS_ENC": bool(args.use_intersection_pos_enc),
        "INTERSECTION_POS_DIM": int(args.intersection_pos_dim),
        "USE_FEATURE_GROUP_GATE": bool(args.use_feature_group_gate),
        "USE_FEATURE_GROUP_CONCAT": bool(args.use_feature_group_concat),
        "FEATURE_GROUP_HIDDEN_DIM": int(args.feature_group_hidden_dim),
        "USE_AUXILIARY_HEAD": bool(args.use_auxiliary_head),
        "AUXILIARY_TASK": str(args.auxiliary_task),
        "AUXILIARY_WEIGHT": float(args.auxiliary_weight),
        "AUXILIARY_EMA_TAU": float(args.auxiliary_ema_tau),
        "USE_TRANSFORMER_ENCODER": True,
        "USE_GAT_AGG": True,
        "TRANS_DIM": int(args.trans_dim),
        "TRANS_HEADS": int(args.trans_heads),
        "TRANS_LAYERS": int(args.trans_layers),
        "TRANS_FFN_DIM": int(args.trans_ffn_dim),
        "TRANS_DROPOUT": float(args.trans_dropout),
        "TRANS_USE_COS_MASK": not bool(args.disable_trans_cos_mask),
        "TRANS_PRENORM": not bool(args.disable_trans_prenorm),
        "USE_BLOCK_ATTN_RES": bool(args.use_block_attn_res),
        "USE_ADAPTIVE_PRESSURE_PHASE_HEAD": False,
        "USE_LIGHT_PHASE_RELATION": False,
        "USE_LIGHT_TEMPORAL_DELTA": False,
        "APL_MOVE_HIDDEN_DIM": 16,
        "APL_PHASE_HIDDEN_DIM": 16,
        "APL_REL_DIM": 16,
        "APL_TEMPORAL_DELTA_WEIGHT": 0.25,
    }

    # Apply ablation overrides with minimal and explicit scope.
    mode = args.ablation_mode
    if mode == "full":
        pass
    elif mode == "no_trans":
        dic_agent_conf_extra["USE_TRANSFORMER_ENCODER"] = False
        dic_agent_conf_extra["USE_GAT_AGG"] = True
    elif mode == "mlp_only":
        dic_agent_conf_extra["USE_TRANSFORMER_ENCODER"] = False
        dic_agent_conf_extra["USE_GAT_AGG"] = False
    elif mode == "std_trans":
        dic_agent_conf_extra["USE_TRANSFORMER_ENCODER"] = True
        dic_agent_conf_extra["TRANS_USE_COS_MASK"] = False
    elif mode == "no_redq":
        dic_agent_conf_extra["USE_TRANSFORMER_ENCODER"] = True
        dic_agent_conf_extra["TRANS_USE_COS_MASK"] = True
        dic_agent_conf_extra["USE_REDQ"] = False
        dic_agent_conf_extra["TRUE_REDQ_MODE"] = False
        dic_agent_conf_extra["REDQ_UTD"] = 1
        dic_agent_conf_extra["REDQ_M"] = 1
        dic_agent_conf_extra["REDQ_PAPER_UTD"] = False
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = False
    elif mode == "mlp_no_redq":
        dic_agent_conf_extra["USE_TRANSFORMER_ENCODER"] = False
        dic_agent_conf_extra["USE_GAT_AGG"] = False
        dic_agent_conf_extra["USE_REDQ"] = False
        dic_agent_conf_extra["TRUE_REDQ_MODE"] = False
        dic_agent_conf_extra["REDQ_M"] = 1
        dic_agent_conf_extra["REDQ_PAPER_UTD"] = False
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = False
    else:
        raise ValueError("Unknown ablation_mode: {}".format(mode))

    # Apply new training-side plan overrides.
    new_plan = args.new_plan
    if new_plan == "A_utd":
        # True UTD=4: now actually executed as 4 gradient update loops per round.
        dic_agent_conf_extra["REDQ_UTD"] = 4
    elif new_plan == "B_soft":
        # Soft Polyak target update every gradient step instead of lagged hard copy.
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "C_epochs":
        # More gradient epochs per round with a lower learning rate.
        dic_agent_conf_extra["EPOCHS"] = 3
        dic_agent_conf_extra["LEARNING_RATE"] = 0.0005
        dic_agent_conf_extra["PATIENCE"] = 3
    elif new_plan == "D_replay":
        # Larger replay buffer to provide more diverse off-policy data.
        dic_agent_conf_extra["MAX_MEMORY_LEN"] = 30000
        dic_agent_conf_extra["SAMPLE_SIZE"] = 5000
    elif new_plan == "E_combo":
        # Combination of all A+B+C+D improvements plus lower min epsilon.
        dic_agent_conf_extra["REDQ_UTD"] = 4
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
        dic_agent_conf_extra["EPOCHS"] = 3
        dic_agent_conf_extra["LEARNING_RATE"] = 0.0005
        dic_agent_conf_extra["PATIENCE"] = 3
        dic_agent_conf_extra["MAX_MEMORY_LEN"] = 30000
        dic_agent_conf_extra["SAMPLE_SIZE"] = 5000
        dic_agent_conf_extra["MIN_EPSILON"] = 0.05
    elif new_plan == "F_dueling":
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A).
        dic_agent_conf_extra["USE_DUELING"] = True
    elif new_plan == "G_double":
        # Double DQN: online net selects action, target net evaluates.
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
    elif new_plan == "H_2step":
        # 2-step return: r_t + γ*r_{t+1}, bootstrap with γ²=0.64, NORMAL_FACTOR adjusted.
        dic_agent_conf_extra["NSTEP"] = 2
        dic_agent_conf_extra["NORMAL_FACTOR"] = 36.0
    elif new_plan == "I_duel_double":
        # Dueling + Double DQN combined.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
    elif new_plan == "J_all":
        # All Rainbow-lite: Dueling + Double + 2-step.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["NSTEP"] = 2
        dic_agent_conf_extra["NORMAL_FACTOR"] = 36.0
    elif new_plan == "K_stable":
        # H_2step + B_soft: stable credit assignment + smooth target updates.
        dic_agent_conf_extra["NSTEP"] = 2
        dic_agent_conf_extra["NORMAL_FACTOR"] = 36.0
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "L_floor":
        # I_duel_double + B_soft: lowest Q-floor + smooth target updates.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "M_ifdg":
        # L_floor + IFDG reward (unbiased ATT proxy, NORMAL_FACTOR=20).
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "N_pressure":
        # L_floor + pressure reward (Advanced-MP style, zero code-change reward).
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "O_ifdg_nf40":
        # L_floor + IFDG reward with larger NORMAL_FACTOR=40 to handle IFDG scale.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
        dic_agent_conf_extra["NORMAL_FACTOR"] = 40.0
    elif new_plan == "P_aplight":
        # L_floor training backbone + adaptive-pressure / phase-relation structured head.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
        dic_agent_conf_extra["USE_ADAPTIVE_PRESSURE_PHASE_HEAD"] = True
        dic_agent_conf_extra["USE_LIGHT_PHASE_RELATION"] = True
        dic_agent_conf_extra["USE_LIGHT_TEMPORAL_DELTA"] = True
    elif new_plan == "P_hybrid":
        # L_floor + queue reward backbone + small auxiliary IFDG term.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "AC_queue_switch":
        # L_floor + pure queue+switch reward, no extra state feature changes.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "AD_queue_switch_fair":
        # L_floor + queue+switch reward + worst-lane queue fairness term.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "AE_queue_switch_balance":
        # L_floor + queue+switch reward + queue-balance regularization.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "Y_queuefair":
        # L_floor + queue reward + worst-lane queue penalty for anti-starvation.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "Z_queuebalance":
        # L_floor + queue reward + queue-balance penalty to discourage local imbalance.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "R_staged_ifdg":
        # L_floor + queue reward first, then gradually blend in a small IFDG term.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan == "AI_qsb_adv":
        # Best reward branch so far: queue + switch + balance, now with advanced pressure supplement.
        dic_agent_conf_extra["USE_DUELING"] = True
        dic_agent_conf_extra["USE_DOUBLE_DQN"] = True
        dic_agent_conf_extra["REDQ_SOFT_TARGET_UPDATE"] = True
        dic_agent_conf_extra["REDQ_TAU"] = 0.01
    elif new_plan != "base":
        raise ValueError("Unknown new_plan: {}".format(new_plan))

    deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)
    if args.static_delay_candidate_mode:
        top_k_adjacency = int(args.static_delay_candidate_rmax)
    elif args.distance_topk_mode:
        top_k_adjacency = int(args.distance_topk_k)
    else:
        top_k_adjacency = int(_max_grid_nodes_within_hop(num_row, num_col, int(args.max_hop)))
    dic_traffic_env_conf_extra = {
        "NUM_ROUNDS": int(args.num_rounds),
        "NUM_GENERATORS": int(args.num_generators),
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": num_row * num_col,
        "RUN_COUNTS": int(args.run_counts),
        "GENERATOR_CPU_ONLY": bool(args.num_generators > 1),
        "GENERATOR_CUDA_VISIBLE_DEVICES": args.generator_cuda_visible_devices,
        "MULTIPROCESS_UPDATER": False,
        "MODEL_NAME": "REDQ",
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": "roadnet_{}.json".format(road_net),
        "TOP_K_ADJACENCY": top_k_adjacency,
        "ADJ_MASK_BY_HOP": bool(int(args.max_hop) >= 0),
        "MAX_HOP_DISTANCE": int(args.max_hop),
        "DISTANCE_TOPK_MODE": bool(args.distance_topk_mode),
        "DISTANCE_TOPK_K": int(args.distance_topk_k),
        "STATIC_DELAY_CANDIDATE_MODE": bool(args.static_delay_candidate_mode),
        "STATIC_DELAY_MULTIPLIER": float(args.static_delay_multiplier),
        "STATIC_DELAY_CANDIDATE_RMAX": int(args.static_delay_candidate_rmax),
        "STATIC_DELAY_MIN_EXTERNAL": int(args.static_delay_min_external),
        "STATIC_DELAY_USE_SHORTEST_PATH": True,
        "STATIC_DELAY_PADDING": "self",
        "MASK_FARTHEST_COUNT": max(0, int(args.mask_farthest_count)),
        "PHASE_ELAPSED_NORM_BASE": float(args.phase_elapsed_norm_base),
        "TREND_WINDOW_STEPS": int(config.dic_traffic_env_conf.get("MIN_ACTION_TIME", 15)),
        "EMA_ALPHA": float(args.ema_alpha),
        "KALMAN_PRESSURE_Q": float(args.kalman_pressure_q),
        "KALMAN_PRESSURE_R": float(args.kalman_pressure_r),
        "SEED": int(args.seed),
        "seed": int(args.seed),
        "NEW_PLAN": str(new_plan),
        "GAMMA": float(args.gamma),
        "NSTEP": int(deploy_dic_agent_conf.get("NSTEP", 1)),
        "saveReplay": True,
        "LIST_STATE_FEATURE": list_state_feature,
        "DIC_REWARD_INFO": _build_reward_info(args.reward_type, new_plan, args),
        "USE_LOGGED_REWARD": False,
        "INTERSECTION_TOPOLOGY_DIM": 8,
    }
    if new_plan == "R_staged_ifdg":
        dic_traffic_env_conf_extra["REWARD_SCHEDULE_MODE"] = "staged_ifdg"
        dic_traffic_env_conf_extra["REWARD_WARMUP_ROUNDS"] = 20
        dic_traffic_env_conf_extra["REWARD_RAMP_ROUNDS"] = 20

    if resume_run_dir:
        dic_traffic_env_conf_extra["RESUME"] = True

    if resume_run_dir:
        if not os.path.isabs(resume_run_dir):
            resume_run_dir = os.path.join(repo_root, resume_run_dir)
        if resume_model_dir is not None and not os.path.isabs(resume_model_dir):
            resume_model_dir = os.path.join(repo_root, resume_model_dir)
        dic_path_extra = {
            "PATH_TO_MODEL": resume_model_dir,
            "PATH_TO_WORK_DIRECTORY": resume_run_dir,
            "PATH_TO_DATA": os.path.join(repo_root, "data", template, road_net),
            "PATH_TO_ERROR": os.path.join(repo_root, "errors", memo),
            "PATH_TO_INIT_MODEL": None,
        }
    else:
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join(repo_root, "model", memo, exp_id),
            "PATH_TO_WORK_DIRECTORY": os.path.join(repo_root, "records", memo, exp_id),
            "PATH_TO_DATA": os.path.join(repo_root, "data", template, road_net),
            "PATH_TO_ERROR": os.path.join(repo_root, "errors", memo),
            "PATH_TO_INIT_MODEL": None,
        }

    dic_traffic_env_conf_extra["MODEL_NAME"] = str(args.model_name)
    deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
    deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

    print("=" * 90)
    print("true-REDQ + Trans CUDA launch")
    print("tf_version={}".format(tf_version))
    print("visible_gpus={}".format([g.name for g in gpus]))
    print("dataset={}, traffic={}, seed={}, rounds={}".format(args.dataset, traffic_file, args.seed, args.num_rounds))
    print("num_generators={}".format(args.num_generators))
    print(
        "REDQ N/M/UTD/lambda={}/{}/{}/{}".format(
            args.redq_n, args.redq_m, args.redq_utd, args.redq_lambda
        )
    )
    print("ablation_mode={}".format(mode))
    print("new_plan={}".format(new_plan))
    print("feature_set={}".format(args.feature_set))
    print(
        "hop_mask={}, max_hop={}, distance_topk={}, distance_k={}, static_delay_mode={}, delay_mult={}, delay_rmax={}, delay_min_ext={}, mask_farthest={}, top_k={}".format(
            bool(deploy_dic_traffic_env_conf.get("ADJ_MASK_BY_HOP")),
            deploy_dic_traffic_env_conf.get("MAX_HOP_DISTANCE"),
            bool(deploy_dic_traffic_env_conf.get("DISTANCE_TOPK_MODE")),
            deploy_dic_traffic_env_conf.get("DISTANCE_TOPK_K"),
            bool(deploy_dic_traffic_env_conf.get("STATIC_DELAY_CANDIDATE_MODE")),
            deploy_dic_traffic_env_conf.get("STATIC_DELAY_MULTIPLIER"),
            deploy_dic_traffic_env_conf.get("STATIC_DELAY_CANDIDATE_RMAX"),
            deploy_dic_traffic_env_conf.get("STATIC_DELAY_MIN_EXTERNAL"),
            deploy_dic_traffic_env_conf.get("MASK_FARTHEST_COUNT"),
            deploy_dic_traffic_env_conf.get("TOP_K_ADJACENCY"),
        )
    )
    print(
        "feature_group_gate={}, feature_group_concat={}, fg_hidden_dim={}".format(
            bool(args.use_feature_group_gate),
            bool(args.use_feature_group_concat),
            int(args.feature_group_hidden_dim),
        )
    )
    print(
        "dynamic_collab={}, cos_enabled={}, mlp_neighbor_agg={}, cos_total_k={}, cos_adj_mode={}, cos_slot_min_prob={}, cos_use_candidate_mask={}".format(
            bool(args.use_dynamic_collab),
            bool(args.enable_cos),
            bool(args.use_mlp_neighbor_agg),
            int(args.cos_total_k),
            str(args.cos_adj_mode),
            float(args.cos_slot_min_prob),
            bool(args.cos_use_input_candidate_mask),
        )
    )
    print(
        "neighbor_select={}, hop={}, topk={}, gate={}, thr={}, temp={}, topo={}, delay={}, same_corridor={}, distance_only={}, rel_hidden={}, state_rel_mode={}".format(
            bool(args.neighbor_select_enabled),
            int(args.neighbor_candidate_hop),
            int(args.neighbor_topk),
            str(args.neighbor_gate_type),
            float(args.neighbor_gate_threshold),
            float(args.neighbor_gate_temp),
            bool(args.use_topo_feature),
            bool(args.use_delay_feature),
            bool(args.use_same_corridor_feature),
            bool(args.delay_use_distance_only),
            int(args.relation_hidden_dim),
            str(args.neighbor_state_rel_mode),
        )
    )
    print(
        "dynamic_collab_full={}, pair_dim={}, need_bias={}".format(
            bool(args.use_dynamic_collab_full),
            int(args.dynamic_collab_pair_dim),
            float(args.dynamic_collab_need_bias),
        )
    )
    print(
        "extra_isr=False, offline_mode=False, finetune_init_model_dir=None"
    )
    print("ema_mode={}, ema_alpha={}".format(args.ema_mode, args.ema_alpha))
    print(
        "use_kalman_pressure={}, kalman_q={}, kalman_r={}".format(
            bool(args.use_kalman_pressure), args.kalman_pressure_q, args.kalman_pressure_r
        )
    )
    print(
        "resolved: redq={}, true_redq={}, utd={}, m={}, trans={}, trans_mask={}".format(
            deploy_dic_agent_conf.get("USE_REDQ"),
            deploy_dic_agent_conf.get("TRUE_REDQ_MODE"),
            deploy_dic_agent_conf.get("REDQ_UTD"),
            deploy_dic_agent_conf.get("REDQ_M"),
            deploy_dic_agent_conf.get("USE_TRANSFORMER_ENCODER"),
            deploy_dic_agent_conf.get("TRANS_USE_COS_MASK"),
        )
    )
    print(
        "REDQ paper_utd={}, soft_target={}, tau={}".format(
            bool(deploy_dic_agent_conf.get("REDQ_PAPER_UTD")),
            bool(deploy_dic_agent_conf.get("REDQ_SOFT_TARGET_UPDATE")),
            float(deploy_dic_agent_conf.get("REDQ_TAU")),
        )
    )
    print(
        "critic: droq_mode={}, ln={}, dropout={}, hidden_dim={}, layers={}".format(
            bool(deploy_dic_agent_conf.get("DROQ_MODE", False)),
            bool(deploy_dic_agent_conf.get("CRITIC_USE_LAYER_NORM", False)),
            float(deploy_dic_agent_conf.get("CRITIC_DROPOUT_RATE", 0.0)),
            int(deploy_dic_agent_conf.get("CRITIC_HIDDEN_DIM", 32)),
            int(deploy_dic_agent_conf.get("CRITIC_NUM_LAYERS", 2)),
        )
    )
    print(
        "crossq_safe={}, bn_mode={}, use_bn={}, joint_forward={}, custom_step={}, keep_target={}, soft_target={}".format(
            bool(deploy_dic_agent_conf.get("CROSSQ_SAFE_MODE", False)),
            str(deploy_dic_agent_conf.get("CROSSQ_BN_MODE", "brn")),
            bool(deploy_dic_agent_conf.get("CROSSQ_USE_BATCH_NORM", False)),
            bool(deploy_dic_agent_conf.get("CROSSQ_JOINT_FORWARD", False)),
            bool(deploy_dic_agent_conf.get("CROSSQ_CUSTOM_TRAIN_STEP", False)),
            bool(deploy_dic_agent_conf.get("CROSSQ_KEEP_TARGET_NET", True)),
            bool(deploy_dic_agent_conf.get("REDQ_SOFT_TARGET_UPDATE", False)),
        )
    )
    print("gamma={}, normal_factor={}, batch_size={}".format(
        deploy_dic_agent_conf.get("GAMMA"),
        deploy_dic_agent_conf.get("NORMAL_FACTOR"),
        deploy_dic_agent_conf.get("BATCH_SIZE")))
    print(
        "PER enabled={}, alpha={}, eps={}, uniform_mix={}, pool_mult={}".format(
            bool(deploy_dic_agent_conf.get("USE_PER")),
            deploy_dic_agent_conf.get("PER_ALPHA"),
            deploy_dic_agent_conf.get("PER_EPS"),
            deploy_dic_agent_conf.get("PER_UNIFORM_MIX"),
            deploy_dic_agent_conf.get("PER_POOL_MULT"),
        )
    )
    print(
        "NoisyNet enabled={}, sigma_init={}".format(
            bool(deploy_dic_agent_conf.get("USE_NOISY_NET")),
            deploy_dic_agent_conf.get("NOISY_SIGMA_INIT"),
        )
    )
    print(
        "UCB enabled={}, lambda={}, decay={}, min={}".format(
            bool(deploy_dic_agent_conf.get("USE_UCB_ACTION")),
            deploy_dic_agent_conf.get("UCB_LAMBDA"),
            deploy_dic_agent_conf.get("UCB_DECAY"),
            deploy_dic_agent_conf.get("UCB_MIN"),
        )
    )
    print(
        "ActionGaussian enabled={}, std={}, clip={}".format(
            float(deploy_dic_agent_conf.get("ACTION_GAUSSIAN_STD", 0.0)) > 0.0,
            deploy_dic_agent_conf.get("ACTION_GAUSSIAN_STD"),
            deploy_dic_agent_conf.get("ACTION_GAUSSIAN_CLIP"),
        )
    )

    print(
        "Trans dim/heads/layers/ffn/dropout={}/{}/{}/{}/{}".format(
            args.trans_dim, args.trans_heads, args.trans_layers, args.trans_ffn_dim, args.trans_dropout
        )
    )
    print("record_dir={}".format(deploy_dic_path["PATH_TO_WORK_DIRECTORY"]))
    print("model_dir={}".format(deploy_dic_path["PATH_TO_MODEL"]))
    print("resume_mode={}".format(bool(resume_run_dir)))
    print("state_features={}".format(deploy_dic_traffic_env_conf["LIST_STATE_FEATURE"]))
    print("=" * 90)

    if args.validate_only:
        print("validate_only=True, skip training launch.")
        return

    ppl = Pipeline(
        dic_agent_conf=deploy_dic_agent_conf,
        dic_traffic_env_conf=deploy_dic_traffic_env_conf,
        dic_path=deploy_dic_path,
    )
    ppl.run(multi_process=bool(args.num_generators > 1))


if __name__ == "__main__":
    main()
