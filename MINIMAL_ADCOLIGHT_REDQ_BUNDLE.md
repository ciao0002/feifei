# Minimal AdvancedCoLight + REDQ Dev Bundle

This bundle is for pulling on another machine to continue development with the current AdvancedCoLight and true-REDQ code paths.

## Included Code

- `run_advanced_colight.py`
- `run_adcolight_jnreal_variants.py`
- `run_adcolight_hz5816_variants.py`
- `run_trueredq_trans_cuda.py`
- `summary.py`
- `models/`
- `utils/` (includes `env_factory.py` and `sumo_env.py`)

## Included Data (minimal subset)

- `data/Jinan/3_4` (jnreal / jn2000 / jn2500)
- `data/Hangzhou/4_4` (hzreal / hz5816)
- `data/Arterial/1_6` (arterial1x6)
- `data/Hangzhou/1_1` (hz1x1)

## Notes

- Training artifacts (`records/`, `model/`, `logs/`, `summary/`) are intentionally excluded by `.gitignore`.
- This commit is intended as a portable development snapshot, not a full experiment archive.
