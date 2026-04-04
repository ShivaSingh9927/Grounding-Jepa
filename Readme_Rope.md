# Baseline (no RoPE)
CUDA_VISIBLE_DEVICES=0 python train.py --run_name my_run
# With RoPE (theta=10000)
CUDA_VISIBLE_DEVICES=0 python train.py --use_rope --run_name my_run
# With larger theta
CUDA_VISIBLE_DEVICES=0 python train.py --use_rope --rope_theta 500000 --run_name my_run