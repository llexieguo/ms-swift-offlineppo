#!/bin/bash
# ============================================================
# OPSD Training Script From PPO-Style Samples
# Usage: bash examples/train/rlhf/opsd/full_from_ppo.sh
# ============================================================

# ======================== 按需修改区 ========================

# 模型
MODEL="Qwen/Qwen3-4B"

# 数据集路径（jsonl 文件）
DATASET="/path/to/your/ppo_samples.jsonl"

# 输出目录
OUTPUT_DIR="output/opsd_from_ppo"

# GPU 设置
CUDA_VISIBLE_DEVICES="0"
NPROC_PER_NODE=1

# 训练模式: "lora" 或 "full"
TUNER_TYPE="full"

# 训练超参
MAX_STEPS=1000
LEARNING_RATE=2e-5
BATCH_SIZE=4
GRAD_ACCUM=1
MAX_LENGTH=8192
MAX_COMPLETION_LENGTH=2048

# LoRA 参数（仅 TUNER_TYPE=lora 时生效）
LORA_RANK=64
LORA_ALPHA=128

# PPO 风格数据 -> OPSD
ANSWER_KEY="answer"
JUDGE_KEY="expect_acc"
JUDGE_THRESHOLD=0.5
# 可选：自定义 teacher prompt 模板，可用 {prompt} 和 {answer}
# TEACHER_PROMPT='{prompt}\n\nCandidate answer:\n{answer}\n\nVerify it and produce your own reasoning.'

# GKD / OPSD 专有
TEACHER_MODEL="${MODEL}"
LMBDA=1.0
BETA=0.5
TEMPERATURE=1.2
SFT_ALPHA=0

# vLLM / rollout
USE_VLLM=true
VLLM_MODE="colocate"
VLLM_GPU_MEMORY_UTILIZATION=0.7
VLLM_MAX_MODEL_LEN=10240
SLEEP_LEVEL=1

# 保存 & 日志
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=10
LOGGING_STEPS=1
REPORT_TO="tensorboard swanlab"
export WANDB_PROJECT="${WANDB_PROJECT:-opsd-from-ppo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ======================== 构建命令 ========================

ARGS=(
    --rlhf_type gkd
    --model "${MODEL}"
    --dataset "${DATASET}"
    --output_dir "${OUTPUT_DIR}"
    --teacher_model "${TEACHER_MODEL}"
    --tuner_type "${TUNER_TYPE}"
    --torch_dtype bfloat16
    --max_steps ${MAX_STEPS}
    --per_device_train_batch_size ${BATCH_SIZE}
    --gradient_accumulation_steps ${GRAD_ACCUM}
    --learning_rate ${LEARNING_RATE}
    --save_steps ${SAVE_STEPS}
    --save_total_limit ${SAVE_TOTAL_LIMIT}
    --logging_steps ${LOGGING_STEPS}
    --max_length ${MAX_LENGTH}
    --max_completion_length ${MAX_COMPLETION_LENGTH}
    --save_only_model true
    --gradient_checkpointing true
    --deepspeed zero0
    --attn_impl flash_attn
    --lmbda ${LMBDA}
    --beta ${BETA}
    --temperature ${TEMPERATURE}
    --sft_alpha ${SFT_ALPHA}
    --ppo_data_transform opsd
    --ppo_data_answer_key "${ANSWER_KEY}"
    --ppo_data_judge_key "${JUDGE_KEY}"
    --ppo_data_judge_threshold ${JUDGE_THRESHOLD}
    --report_to ${REPORT_TO}
)

if [ "${USE_VLLM}" = "true" ]; then
    ARGS+=(
        --use_vllm true
        --vllm_mode "${VLLM_MODE}"
        --vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION}
        --vllm_max_model_len ${VLLM_MAX_MODEL_LEN}
        --sleep_level ${SLEEP_LEVEL}
    )
fi

if [ -n "${TEACHER_PROMPT:-}" ]; then
    ARGS+=(--ppo_data_teacher_prompt "${TEACHER_PROMPT}")
fi

# LoRA 模式追加参数
if [ "${TUNER_TYPE}" = "lora" ]; then
    ARGS+=(
        --lora_rank ${LORA_RANK}
        --lora_alpha ${LORA_ALPHA}
        --target_modules all-linear
    )
fi

# ======================== 启动训练 ========================

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift rlhf "${ARGS[@]}"
