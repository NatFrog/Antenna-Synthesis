# List presets
python -m scripts.tune_multihead_v4 --list-presets

# Preview a run without executing
python -m scripts.tune_multihead_v4 --preset inference-default --dry-run

# Tier 1: inference sweep on baseline v4 checkpoint (~minutes)
python -m scripts.tune_multihead_v4 --preset inference-default

# Tier 1: α_main cap sweep (includes 16×16 regional — not in stock eval)
python -m scripts.tune_multihead_v4 --preset alpha-main-cap

# Tier 2: retrain + post-hoc eval (hours)
python -m scripts.tune_multihead_v4 --mode train --preset mainbeam-focus --init-v2

# Custom checkpoint
python -m scripts.tune_multihead_v4 --ckpt checkpoints/tuning_v4/mainbeam70_w035/best_generator.pt --preset inference-default
