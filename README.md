# REDQ-MLP + Static Collaborator Review Package

This folder is a review-ready copy of the current method code path used in our experiments.

Scope:
- Current method only: `REDQ` with `ablation_mode=mlp_only`
- Static collaborator selection from cached delay graph
- Delay-aware relational neighbor message mean aggregation
- Original replay buffer, REDQ ensemble critic update, and CityFlow training loop

Main entrypoints:
- `launchers/launch_redqfix_staticdelay1x_relmsgmean_utdwarm10_16to4_6flows_s42.py`

Core files:
- `run_redq_trans.py`
- `models/redq_agent.py`
- `utils/cityflow_env.py`
- `utils/pipeline.py`
- `utils/generator.py`
- `utils/construct_sample.py`
- `utils/updater.py`
- `utils/model_test.py`
- `utils/config.py`

Current method flags used by the main 6-flow launcher:
- `-ablation_mode mlp_only`
- `-feature_set baseline`
- `-redq_n 4`
- `-redq_m 2`
- `-redq_utd 4`
- `-redq_utd_warmup_rounds 10`
- `-redq_utd_warmup_value 16`
- `-redq_utd_after_value 4`
- `-redq_lambda 1.0`
- `-static_delay_candidate_mode`
- `-static_delay_multiplier 1.0`
- `-static_delay_candidate_rmax 8`
- `-static_delay_min_external 0`
- `-use_delay_rel_msg_mean`
- `-delay_msg_hidden_dim 16`
- `-delay_msg_delta_reduce mean`

Method path summary:
1. `run_redq_trans.py` builds config and launches `Pipeline`
2. `utils/generator.py` runs CityFlow and logs transition data
3. `utils/construct_sample.py` converts logs into replay samples
4. `utils/updater.py` loads samples and calls `models/redq_agent.py`
5. `models/redq_agent.py` builds local hidden representation, static collaborator messages, and REDQ critics

This package intentionally removes unrelated model registrations, scan launchers,
and legacy selector modules so code review stays focused on the active method only.
