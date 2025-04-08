export JUDGE_LLM_API_KEY="sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86"
export JUDGE_LLM_API_BASE_URL="http://147.46.91.62:32390/v1"
export JUDGE_LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"

export VLLM_USE_V1=0

export OUTPUT_DIR="./logs/mazenav_vqa"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vllm \
    --model_args model_version="google/gemma-3-4b-it",gpu_memory_utilization=0.95,max_videos=0,max_audios=0,max_images=1 \
    --tasks mazenav_vqa \
    --batch_size 1 \
    --log_samples \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-gridworld,job_type=eval,name="$(basename $OUTPUT_DIR)" \
    # --use_cache ".cache"\
    # --log_samples_suffix llava-ov-7b \