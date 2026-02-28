#!/usr/bin/env python3
"""
CUDA launcher for true-REDQ + Transformer on Advanced_XLight (MHQCoSLight backend).

This runner keeps the algorithm core unchanged and only wraps:
1) GPU visibility / TF runtime setup
2) true-REDQ + Transformer configuration
3) single-dataset training launch (jnreal / hzreal / hz5816)
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
    "newyork": {
        "template": "newyork_28_7",
        "road_net": "28_7",
        "traffic_file": "anon_28_7_newyork_real_double.json",
        "exp_prefix": "newyork_28_7",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default="jnreal", choices=sorted(DATASET_PRESETS.keys()))
    parser.add_argument("-traffic_file", type=str, default=None)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-num_rounds", type=int, default=60)
    parser.add_argument("-run_counts", type=int, default=3600)
    parser.add_argument("-num_generators", type=int, default=1)
    parser.add_argument("-memo_prefix", type=str, default="trueredq_trans_cuda")

    parser.add_argument("-epochs", type=int, default=1)
    parser.add_argument("-batch_size", type=int, default=20)
    parser.add_argument("-sample_size", type=int, default=3000)
    parser.add_argument("-min_epsilon", type=float, default=0.2)

    parser.add_argument("-redq_n", type=int, default=4)
    parser.add_argument("-redq_m", type=int, default=2)
    parser.add_argument("-redq_utd", type=int, default=4)
    parser.add_argument("-redq_lambda", type=float, default=1.0)
    parser.add_argument("-redq_paper_utd", action="store_true",
                        help="Use paper-style UTD: resample replay for every UTD update, including the first one.")
    parser.add_argument("-redq_soft_target", action="store_true",
                        help="Use Polyak soft target updates (paper-style) instead of lagged hard target copy.")
    parser.add_argument("-redq_tau", type=float, default=0.005,
                        help="Polyak coefficient tau for soft target update: target=(1-tau)*target+tau*online.")

    parser.add_argument("-trans_dim", type=int, default=32)
    parser.add_argument("-trans_heads", type=int, default=4)
    parser.add_argument("-trans_layers", type=int, default=2)
    parser.add_argument("-trans_ffn_dim", type=int, default=128)
    parser.add_argument("-trans_dropout", type=float, default=0.1)
    parser.add_argument("-disable_trans_cos_mask", action="store_true")
    parser.add_argument("-disable_trans_prenorm", action="store_true")

    parser.add_argument("-add_phase_elapsed", action="store_true")
    parser.add_argument("-add_downstream_congestion", action="store_true")
    parser.add_argument("-tsa_enable", action="store_true")
    parser.add_argument("-tsa_gaussian_std", type=float, default=0.0)
    parser.add_argument("-tsa_mask_prob", type=float, default=0.0)
    parser.add_argument("-tsa_scale_low", type=float, default=1.0)
    parser.add_argument("-tsa_scale_high", type=float, default=1.0)
    parser.add_argument("-disable_tsa_on_next_state", action="store_true")

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
    if args.num_generators > 1:
        # Avoid forking a process with an already-initialized TF CUDA runtime.
        from multiprocessing import set_start_method
        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    tf_version, gpus = configure_tf_runtime(args)

    preset = DATASET_PRESETS[args.dataset]
    road_net = preset["road_net"]
    traffic_file = args.traffic_file or preset["traffic_file"]
    template = preset["template"]
    num_row = int(road_net.split("_")[0])
    num_col = int(road_net.split("_")[1])

    from utils import config
    from utils.utils import merge
    from utils.pipeline import Pipeline

    memo = "{}_{}_s{}_n{}m{}u{}".format(
        args.memo_prefix, args.dataset, args.seed, args.redq_n, args.redq_m, args.redq_utd
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

    list_state_feature = [
        "cur_phase",
        "traffic_movement_pressure_queue_efficient",
        "lane_enter_running_part",
    ]
    if args.add_phase_elapsed:
        list_state_feature.append("phase_elapsed")
    if args.add_downstream_congestion:
        list_state_feature.append("downstream_congestion")
    list_state_feature.append("adjacency_matrix")

    dic_agent_conf_extra = {
        "CNN_layers": [[32, 32]],
        "USE_MULTIHEAD_Q": False,
        "HEAD_N": int(args.redq_n),
        "HEAD_AGG": "mean",
        "HEAD_DEBUG": False,
        "USE_UCB_ACTION": False,
        "USE_HEAD_BOOTSTRAP": False,
        "MIN_EPSILON": float(args.min_epsilon),
        "EPOCHS": int(args.epochs),
        "BATCH_SIZE": int(args.batch_size),
        "SAMPLE_SIZE": int(args.sample_size),
        "USE_REDQ": True,
        "TRUE_REDQ_MODE": True,
        "REDQ_N": int(args.redq_n),
        "REDQ_M": int(args.redq_m),
        "REDQ_UTD": int(args.redq_utd),
        "REDQ_LAMBDA": float(args.redq_lambda),
        "REDQ_PAPER_UTD": bool(args.redq_paper_utd),
        "REDQ_SOFT_TARGET_UPDATE": bool(args.redq_soft_target),
        "REDQ_TAU": float(args.redq_tau),
        "RELIGHT_ACTION_VOTE": False,
        "COS_ENABLED": False,
        "USE_TRANSFORMER_ENCODER": True,
        "TRANS_DIM": int(args.trans_dim),
        "TRANS_HEADS": int(args.trans_heads),
        "TRANS_LAYERS": int(args.trans_layers),
        "TRANS_FFN_DIM": int(args.trans_ffn_dim),
        "TRANS_DROPOUT": float(args.trans_dropout),
        "TRANS_USE_COS_MASK": not bool(args.disable_trans_cos_mask),
        "TRANS_PRENORM": not bool(args.disable_trans_prenorm),
        "TSA_ENABLED": bool(args.tsa_enable),
        "TSA_GAUSSIAN_STD": float(args.tsa_gaussian_std),
        "TSA_MASK_PROB": float(args.tsa_mask_prob),
        "TSA_SCALE_LOW": float(args.tsa_scale_low),
        "TSA_SCALE_HIGH": float(args.tsa_scale_high),
        "TSA_APPLY_TO_NEXT_STATE": not bool(args.disable_tsa_on_next_state),
    }
    deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)

    dic_traffic_env_conf_extra = {
        "NUM_ROUNDS": int(args.num_rounds),
        "NUM_GENERATORS": int(args.num_generators),
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": num_row * num_col,
        "RUN_COUNTS": int(args.run_counts),
        "GENERATOR_CPU_ONLY": bool(args.num_generators > 1),
        "MULTIPROCESS_UPDATER": False,
        "MODEL_NAME": "MHQCoSLight",
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": "roadnet_{}.json".format(road_net),
        "TOP_K_ADJACENCY": 5,
        "SEED": int(args.seed),
        "seed": int(args.seed),
        "saveReplay": True,
        "LIST_STATE_FEATURE": list_state_feature,
        "DIC_REWARD_INFO": {
            "queue_length": -0.25,
        },
    }
    if resume_run_dir:
        dic_traffic_env_conf_extra["RESUME"] = True

    if resume_run_dir:
        dic_path_extra = {
            "PATH_TO_MODEL": resume_model_dir,
            "PATH_TO_WORK_DIRECTORY": resume_run_dir,
            "PATH_TO_DATA": os.path.join("data", template, road_net),
            "PATH_TO_ERROR": os.path.join("errors", memo),
        }
    else:
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, exp_id),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, exp_id),
            "PATH_TO_DATA": os.path.join("data", template, road_net),
            "PATH_TO_ERROR": os.path.join("errors", memo),
        }

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
    print(
        "REDQ paper_utd={}, soft_target={}, tau={}".format(
            bool(args.redq_paper_utd), bool(args.redq_soft_target), float(args.redq_tau)
        )
    )
    print(
        "Trans dim/heads/layers/ffn/dropout={}/{}/{}/{}/{}".format(
            args.trans_dim, args.trans_heads, args.trans_layers, args.trans_ffn_dim, args.trans_dropout
        )
    )
    print(
        "TSA enabled={}, gaussian_std={}, mask_prob={}, scale_range=[{}, {}], apply_to_next_state={}".format(
            bool(args.tsa_enable),
            args.tsa_gaussian_std,
            args.tsa_mask_prob,
            args.tsa_scale_low,
            args.tsa_scale_high,
            (not bool(args.disable_tsa_on_next_state)),
        )
    )
    print("record_dir={}".format(deploy_dic_path["PATH_TO_WORK_DIRECTORY"]))
    print("model_dir={}".format(deploy_dic_path["PATH_TO_MODEL"]))
    print("resume_mode={}".format(bool(resume_run_dir)))
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
