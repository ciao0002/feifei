#!/usr/bin/env python3
"""
Run AdvancedCoLight-framework variants on Jinan real:
1) +CoS
2) +Trans
3) +CoS+Trans

Implementation uses MHQCoSLight model backend with:
- single Q head (no MHQ, no REDQ)
- AdvancedCoLight state/reward settings
- toggled CoS/Transformer modules per variant
"""

import argparse
import os
import time

from utils import config
from utils.utils import merge, pipeline_wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-variant", type=str, required=True, choices=["cos", "trans", "cos_trans"])
    parser.add_argument("-memo_prefix", type=str, default="adcolight_jnreal_s42_60r")
    parser.add_argument("-num_rounds", type=int, default=60)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-traffic_file", type=str, default="anon_3_4_jinan_real.json")
    parser.add_argument("-cos_total_k", type=int, default=5)
    parser.add_argument("-cos_adj_mode", type=str, default="tiled_sparse", choices=["tiled_sparse", "topk_slots"])
    parser.add_argument("-cos_slot_min_prob", type=float, default=0.0)
    parser.add_argument("-beta_diag", type=float, default=0.0)
    parser.add_argument("-gamma_sym", type=float, default=0.0)
    parser.add_argument("-entropy_coef", type=float, default=0.0)
    parser.add_argument("-cos_temporal_smooth_coef", type=float, default=0.0)
    parser.add_argument("-cos_budget_coef", type=float, default=0.0)
    parser.add_argument("-cos_budget_thr", type=float, default=0.0)
    parser.add_argument("-cos_budget_tau", type=float, default=0.05)
    parser.add_argument("-cos_explore_mode", type=str, default="none", choices=["none", "gumbel_topk"])
    parser.add_argument("-cos_explore_prob", type=float, default=0.0)
    parser.add_argument("-cos_gumbel_tau", type=float, default=1.0)
    parser.add_argument("-cos_gumbel_scale", type=float, default=1.0)
    parser.add_argument("-cos_explore_infer", action="store_true")
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=20)
    parser.add_argument("-sample_size", type=int, default=3000)
    parser.add_argument("-min_epsilon", type=float, default=0.2)
    parser.add_argument("-trans_dim", type=int, default=32)
    parser.add_argument("-trans_heads", type=int, default=4)
    parser.add_argument("-trans_layers", type=int, default=2)
    parser.add_argument("-trans_ffn_dim", type=int, default=128)
    parser.add_argument("-trans_dropout", type=float, default=0.1)
    parser.add_argument("-disable_trans_cos_mask", action="store_true")
    parser.add_argument("-disable_trans_prenorm", action="store_true")
    parser.add_argument("-use_relight", action="store_true")
    parser.add_argument(
        "-use_true_redq",
        action="store_true",
        help="Enable true REDQ ensemble without enabling RELight action-vote behavior.",
    )
    parser.add_argument("-relight_n", type=int, default=5)
    parser.add_argument("-relight_m", type=int, default=2)
    parser.add_argument("-relight_utd", type=int, default=4)
    parser.add_argument("-disable_relight_vote", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    use_cos = args.variant in ("cos", "cos_trans")
    use_trans = args.variant in ("trans", "cos_trans")

    road_net = "3_4"
    template = "Jinan"
    num_row = int(road_net.split("_")[0])
    num_col = int(road_net.split("_")[1])

    memo = "{}_{}".format(args.memo_prefix, args.variant)
    exp_id = "mhq_coslight_jn_real_{}".format(
        time.strftime("%m%d_%H%M%S", time.localtime(time.time()))
    )

    use_relight = bool(args.use_relight)
    use_true_redq = bool(args.use_true_redq or use_relight)

    dic_agent_conf_extra = {
        "CNN_layers": [[32, 32]],
        # Keep AdvancedCoLight-like state/reward setup; Q-backend can switch to true-REDQ.
        "USE_MULTIHEAD_Q": False,
        "USE_REDQ": bool(use_true_redq),
        "TRUE_REDQ_MODE": bool(use_true_redq),
        "HEAD_N": int(args.relight_n) if use_true_redq else 1,
        "REDQ_N": int(args.relight_n),
        "REDQ_M": int(args.relight_m),
        "REDQ_UTD": int(args.relight_utd),
        "REDQ_LAMBDA": 1.0,
        "RELIGHT_ACTION_VOTE": bool(use_relight and (not args.disable_relight_vote)),
        "HEAD_AGG": "mean",
        "HEAD_DEBUG": False,
        "USE_UCB_ACTION": False,
        "USE_HEAD_BOOTSTRAP": False,
        "MIN_EPSILON": args.min_epsilon,
        "EPOCHS": int(args.epochs),
        "BATCH_SIZE": int(args.batch_size),
        "SAMPLE_SIZE": int(args.sample_size),
        # CoS (on/off by variant)
        "COS_ENABLED": bool(use_cos),
        "COS_TOTAL_K": int(args.cos_total_k),
        "COS_INCLUDE_SELF": True,
        "COS_ADJ_MODE": args.cos_adj_mode,
        "COS_SLOT_MIN_PROB": float(args.cos_slot_min_prob),
        "COS_BETA_DIAG": float(args.beta_diag),
        "COS_GAMMA_SYM": float(args.gamma_sym),
        "COS_ENTROPY_COEF": float(args.entropy_coef),
        "COS_TEMPORAL_SMOOTH_COEF": float(args.cos_temporal_smooth_coef),
        "COS_BUDGET_COEF": float(args.cos_budget_coef),
        "COS_BUDGET_THR": float(args.cos_budget_thr),
        "COS_BUDGET_TAU": float(args.cos_budget_tau),
        "COS_EXPLORE_MODE": args.cos_explore_mode,
        "COS_EXPLORE_PROB": float(args.cos_explore_prob),
        "COS_GUMBEL_TAU": float(args.cos_gumbel_tau),
        "COS_GUMBEL_SCALE": float(args.cos_gumbel_scale),
        "COS_EXPLORE_INFER": bool(args.cos_explore_infer),
        # Transformer (on/off by variant)
        "USE_TRANSFORMER_ENCODER": bool(use_trans),
        "TRANS_DIM": int(args.trans_dim),
        "TRANS_HEADS": int(args.trans_heads),
        "TRANS_LAYERS": int(args.trans_layers),
        "TRANS_FFN_DIM": int(args.trans_ffn_dim),
        "TRANS_DROPOUT": float(args.trans_dropout),
        "TRANS_USE_COS_MASK": not bool(args.disable_trans_cos_mask),
        "TRANS_PRENORM": not bool(args.disable_trans_prenorm),
    }
    deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)

    dic_traffic_env_conf_extra = {
        "NUM_ROUNDS": int(args.num_rounds),
        "NUM_GENERATORS": 1,
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": num_row * num_col,
        "RUN_COUNTS": 3600,
        "MODEL_NAME": "MHQCoSLight",
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "TRAFFIC_FILE": args.traffic_file,
        "ROADNET_FILE": "roadnet_{}.json".format(road_net),
        "TOP_K_ADJACENCY": int(args.cos_total_k),
        "SEED": int(args.seed),
        "seed": int(args.seed),
        "saveReplay": True,
        # Keep AdvancedCoLight state config.
        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "lane_enter_running_part",
            "adjacency_matrix",
        ],
        "DIC_REWARD_INFO": {
            "queue_length": -0.25,
        },
    }

    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", memo, exp_id),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, exp_id),
        "PATH_TO_DATA": os.path.join("data", template, road_net),
        "PATH_TO_ERROR": os.path.join("errors", memo),
    }

    deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
    deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

    print("=" * 90)
    print("Start variant:", args.variant)
    print("traffic={}, seed={}, rounds={}".format(args.traffic_file, args.seed, args.num_rounds))
    print(
        "true_redq_enabled={}, relight_enabled={}, N={}, M={}, UTD={}, vote={}".format(
            use_true_redq,
            use_relight,
            args.relight_n,
            args.relight_m,
            args.relight_utd,
            bool(use_relight and (not args.disable_relight_vote)),
        )
    )
    print("cos_enabled={}, trans_enabled={}".format(use_cos, use_trans))
    print("cos_adj_mode={}, cos_k={}".format(args.cos_adj_mode, args.cos_total_k))
    print(
        "cos_reg diag={}, sym={}, ent={}, smooth={}, budget_coef={}, budget_thr={}, budget_tau={}".format(
            args.beta_diag,
            args.gamma_sym,
            args.entropy_coef,
            args.cos_temporal_smooth_coef,
            args.cos_budget_coef,
            args.cos_budget_thr,
            args.cos_budget_tau,
        )
    )
    print(
        "cos_explore mode={}, prob={}, gumbel_tau={}, gumbel_scale={}, infer={}".format(
            args.cos_explore_mode,
            args.cos_explore_prob,
            args.cos_gumbel_tau,
            args.cos_gumbel_scale,
            bool(args.cos_explore_infer),
        )
    )
    print(
        "transformer dim={}, heads={}, layers={}, ffn_dim={}, dropout={}, mask={}, prenorm={}".format(
            args.trans_dim,
            args.trans_heads,
            args.trans_layers,
            args.trans_ffn_dim,
            args.trans_dropout,
            (not args.disable_trans_cos_mask),
            (not args.disable_trans_prenorm),
        )
    )
    print("record_dir={}".format(deploy_dic_path["PATH_TO_WORK_DIRECTORY"]))
    print("model_dir={}".format(deploy_dic_path["PATH_TO_MODEL"]))
    print("=" * 90)

    pipeline_wrapper(
        dic_agent_conf=deploy_dic_agent_conf,
        dic_traffic_env_conf=deploy_dic_traffic_env_conf,
        dic_path=deploy_dic_path,
    )


if __name__ == "__main__":
    main()
