export JUDGE_LLM_API_KEY="sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86"
export JUDGE_LLM_API_BASE_URL="http://147.46.91.62:32390/v1"
export JUDGE_LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"

export OUTPUT_DIR="./logs/mazenav_group"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks mazenav_vqa,mazenav_vqa_coc \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava-ov-7b \
    --output_path "$OUTPUT_DIR" \
    --wandb_log_samples \
    --wandb_args project=lmms-gridworld,job_type=eval,name="$(basename $OUTPUT_DIR)" \
    # --use_cache ".cache"\