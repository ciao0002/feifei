#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/yanghang/envs/cityflow_47_packed/bin/python"
RUNNER = str(ROOT / "run_redq_trans.py")
LOG_DIR = str(ROOT / "logs")


FLOWS = [
    "jnreal",
    "jn2000",
    "jn2500",
    "hzreal",
    "hz5816",
    "manhattan16x3",
]


def make_cmd(flow: str) -> list[str]:
    memo = f"redqfix_staticdelay1x_relmsgmean_utdwarm10_16to4_{flow}_s42"
    return [
        PYTHON,
        "-u",
        RUNNER,
        "-dataset", flow,
        "-seed", "42",
        "-num_rounds", "60",
        "-memo_prefix", memo,
        "-ablation_mode", "mlp_only",
        "-feature_set", "baseline",
        "-reward_type", "queue",
        "-new_plan", "base",
        "-cuda_visible_devices", "-1",
        "-redq_n", "4",
        "-redq_m", "2",
        "-redq_utd", "4",
        "-redq_lambda", "1.0",
        "-redq_utd_warmup_rounds", "10",
        "-redq_utd_warmup_value", "16",
        "-redq_utd_after_value", "4",
        "-static_delay_candidate_mode",
        "-static_delay_multiplier", "1.0",
        "-static_delay_candidate_rmax", "8",
        "-static_delay_min_external", "0",
        "-use_delay_rel_msg_mean",
        "-delay_msg_hidden_dim", "16",
        "-delay_msg_delta_reduce", "mean",
    ]


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    pid_lines = []

    for flow in FLOWS:
        log_path = os.path.join(
            LOG_DIR,
            f"redqfix_staticdelay1x_relmsgmean_utdwarm10_16to4_{flow}_s42.log",
        )
        cmd = make_cmd(flow)
        with open(log_path, "ab") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        print(f"{flow}: pid={proc.pid} log={log_path}")
        pid_lines.append(f"{flow}\t{proc.pid}\t{log_path}\n")

    pid_path = Path(LOG_DIR) / "redqfix_staticdelay1x_relmsgmean_utdwarm10_16to4_6flows_s42.pids.tsv"
    pid_path.write_text("".join(pid_lines), encoding="utf-8")
    print(f"pid_tsv={pid_path}")


if __name__ == "__main__":
    main()
