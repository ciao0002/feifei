#!/usr/bin/env python3
"""
MHQ-CoSLight ablation line on Hangzhou real 5816 (serial):
Exp1: MHQ + CoS (no diag/sym constraint)
Exp2: MHQ + CoS + diag
Exp3: MHQ + CoS + diag + sym
"""

import argparse
import os
import time

from utils import config
from utils.utils import merge, pipeline_wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo_prefix", type=str, default="mhq_coslight_hz5816_60_s2024")
    parser.add_argument("-num_rounds", type=int, default=60)
    parser.add_argument("-seed", type=int, default=2024)
    parser.add_argument("-head_n", type=int, default=5)
    parser.add_argument("-min_epsilon", type=float, default=0.2)
    parser.add_argument("-traffic_file", type=str, default="anon_4_4_hangzhou_real_5816.json")
    parser.add_argument("-cos_total_k", type=int, default=5)
    parser.add_argument("-beta_diag", type=float, default=0.05)
    parser.add_argument("-gamma_sym", type=float, default=0.10)
    parser.add_argument("-entropy_coef", type=float, default=0.005)
    parser.add_argument("-use_transformer_encoder", action="store_true")
    parser.add_argument("-trans_dim", type=int, default=32)
    parser.add_argument("-trans_heads", type=int, default=4)
    parser.add_argument("-trans_layers", type=int, default=2)
    parser.add_argument("-trans_ffn_dim", type=int, default=128)
    parser.add_argument("-trans_dropout", type=float, default=0.1)
    parser.add_argument("-disable_trans_cos_mask", action="store_true")
    parser.add_argument("-disable_trans_prenorm", action="store_true")
    parser.add_argument(
        "-start_case",
        type=str,
        default="exp1",
        choices=["exp1", "exp2", "exp3"],
        help="Start running from this case (inclusive).",
    )
    parser.add_argument(
        "-only_case",
        type=str,
        default=None,
        choices=["exp1", "exp2", "exp3"],
        help="Run only one case; if set, -start_case is ignored.",
    )
    return parser.parse_args()


def build_case(name, beta_diag, gamma_sym):
    return {
        "name": name,
        "beta_diag": beta_diag,
        "gamma_sym": gamma_sym,
    }


def run_case(args, case):
    traffic_file = args.traffic_file
    road_net = "4_4"
    template = "Hangzhou"
    num_row = int(road_net.split("_")[0])
    num_col = int(road_net.split("_")[1])

    memo = "{}_{}".format(args.memo_prefix, case["name"])
    exp_id = "mhq_coslight_hz_{}_{}".format(
        "5816" if "5816" in traffic_file else "real",
        time.strftime("%m%d_%H%M%S", time.localtime(time.time())),
    )

    dic_agent_conf_extra = {
        "CNN_layers": [[32, 32]],
        # MHQ core
        "USE_MULTIHEAD_Q": True,
        "HEAD_N": args.head_n,
        "HEAD_AGG": "mean",
        "HEAD_DEBUG": False,
        "USE_UCB_ACTION": False,
        "USE_HEAD_BOOTSTRAP": False,
        "MIN_EPSILON": args.min_epsilon,
        # CoS core
        "COS_ENABLED": True,
        "COS_TOTAL_K": args.cos_total_k,
        "COS_INCLUDE_SELF": True,
        "COS_BETA_DIAG": case["beta_diag"],
        "COS_GAMMA_SYM": case["gamma_sym"],
        "COS_ENTROPY_COEF": args.entropy_coef,
        "USE_TRANSFORMER_ENCODER": args.use_transformer_encoder,
        "TRANS_DIM": args.trans_dim,
        "TRANS_HEADS": args.trans_heads,
        "TRANS_LAYERS": args.trans_layers,
        "TRANS_FFN_DIM": args.trans_ffn_dim,
        "TRANS_DROPOUT": args.trans_dropout,
        "TRANS_USE_COS_MASK": not args.disable_trans_cos_mask,
        "TRANS_PRENORM": not args.disable_trans_prenorm,
    }
    deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)

    dic_traffic_env_conf_extra = {
        "NUM_ROUNDS": args.num_rounds,
        "NUM_GENERATORS": 1,
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": num_row * num_col,
        "RUN_COUNTS": 3600,
        "MODEL_NAME": "MHQCoSLight",
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": "roadnet_{}.json".format(road_net),
        "TOP_K_ADJACENCY": args.cos_total_k,
        "SEED": int(args.seed),
        "seed": int(args.seed),
        "saveReplay": True,
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

    print("\n" + "=" * 90)
    print("Start MHQ-CoSLight case: {}".format(case["name"]))
    print("traffic={}, seed={}, rounds={}".format(traffic_file, args.seed, args.num_rounds))
    print("head_n={}, cos_k={}, beta_diag={}, gamma_sym={}, ent_coef={}".format(
        args.head_n, args.cos_total_k, case["beta_diag"], case["gamma_sym"], args.entropy_coef
    ))
    print(
        "transformer_enabled={}, dim={}, heads={}, layers={}, ffn_dim={}, dropout={}, mask={}, prenorm={}".format(
            args.use_transformer_encoder,
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
    print("=" * 90 + "\n")

    pipeline_wrapper(
        dic_agent_conf=deploy_dic_agent_conf,
        dic_traffic_env_conf=deploy_dic_traffic_env_conf,
        dic_path=deploy_dic_path,
    )


def main():
    args = parse_args()
    cases = [
        build_case("exp1", beta_diag=0.0, gamma_sym=0.0),
        build_case("exp2", beta_diag=args.beta_diag, gamma_sym=0.0),
        build_case("exp3", beta_diag=args.beta_diag, gamma_sym=args.gamma_sym),
    ]
    if args.only_case is not None:
        selected = next(c for c in cases if c["name"] == args.only_case)
        run_case(args, selected)
        print("MHQ-CoSLight only_case={} completed.".format(args.only_case))
        return
    start_idx = next(i for i, c in enumerate(cases) if c["name"] == args.start_case)
    for case in cases[start_idx:]:
        run_case(args, case)
    print("All MHQ-CoSLight cases completed.")


if __name__ == "__main__":
    main()
