#!/bin/bash
#SBATCH -J launch
#SBATCH -p gpu-farm
#SBATCH -q high_gpu_users
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err 
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ohjoonhee@snu.ac.kr

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG

export JUDGE_LLM_API_KEY="sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86"
export JUDGE_LLM_API_BASE_URL="http://147.46.91.62:32390/v1"
export JUDGE_LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"

export OUTPUT_DIR="./outputs/mazenav_og_group_vllm"

python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model vllm \
    --model_args model_version="llava-hf/llava-onevision-qwen2-7b-ov-hf",gpu_memory_utilization=0.95 \
    --tasks mazenav_cot,mazenav_coc \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava-ov-7b \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-gridworld,job_type=eval,name="$(basename $OUTPUT_DIR)" \
    # --use_cache ".cache"\