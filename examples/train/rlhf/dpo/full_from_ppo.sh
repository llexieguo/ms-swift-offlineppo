#!/bin/bash
# ============================================================
# DPO Training Script From PPO-Style Samples
# Usage: bash examples/train/rlhf/dpo/full_from_ppo.sh
# ============================================================

# ======================== 按需修改区 ========================

# 模型
MODEL="Qwen/Qwen2.5-7B-Instruct"

# 数据集路径（jsonl 文件）
DATASET="/path/to/your/ppo_samples.jsonl"

# 输出目录
OUTPUT_DIR="output/dpo_from_ppo"

# GPU 设置
CUDA_VISIBLE_DEVICES="0"
NPROC_PER_NODE=1

# 训练模式: "lora" 或 "full"
TUNER_TYPE="full"

# 训练超参
NUM_EPOCHS=1
LEARNING_RATE=1e-4
BATCH_SIZE=1
GRAD_ACCUM=16
MAX_LENGTH=2048

# LoRA 参数（仅 TUNER_TYPE=lora 时生效）
LORA_RANK=8
LORA_ALPHA=32

# PPO 风格数据 -> DPO 偏好对
ANSWER_KEY="answer"
# 同一个 prompt 下，按以下加权分数排序，最高分做 chosen，最低分做 rejected
# 例如: expect_acc + 0.1 * llm_score
SCORE_KEYS="expect_acc,llm_score"
SCORE_WEIGHTS="1,0.1"

# DPO 专有
RPO_ALPHA=0.1

# 保存 & 日志
SAVE_STRATEGY="steps"
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=5
EVAL_RATIO=0.01

# 日志平台: "tensorboard" 或 "wandb"（可同时用: "tensorboard wandb"）
REPORT_TO="tensorboard"
export WANDB_PROJECT="${WANDB_PROJECT:-dpo-from-ppo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ======================== 构建命令 ========================

ARGS=(
    --rlhf_type dpo
    --model "${MODEL}"
    --dataset "${DATASET}"
    --output_dir "${OUTPUT_DIR}"
    --tuner_type "${TUNER_TYPE}"
    --torch_dtype bfloat16
    --num_train_epochs ${NUM_EPOCHS}
    --per_device_train_batch_size ${BATCH_SIZE}
    --per_device_eval_batch_size ${BATCH_SIZE}
    --learning_rate ${LEARNING_RATE}
    --lr_scheduler_type cosine
    --gradient_accumulation_steps ${GRAD_ACCUM}
    --gradient_checkpointing true
    --max_length ${MAX_LENGTH}
    --warmup_ratio 0.05
    --save_strategy ${SAVE_STRATEGY}
    --save_steps ${SAVE_STEPS}
    --save_total_limit ${SAVE_TOTAL_LIMIT}
    --save_only_model true
    --logging_steps ${LOGGING_STEPS}
    --split_dataset_ratio ${EVAL_RATIO}
    --eval_strategy ${SAVE_STRATEGY}
    --eval_steps ${SAVE_STEPS}
    --dataloader_num_workers 4
    --dataset_num_proc 4
    --load_from_cache_file true
    --rpo_alpha ${RPO_ALPHA}
    --ppo_data_transform dpo
    --ppo_data_answer_key "${ANSWER_KEY}"
    --ppo_data_score_keys "${SCORE_KEYS}"
    --ppo_data_score_weights "${SCORE_WEIGHTS}"
    --report_to ${REPORT_TO}
)

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
