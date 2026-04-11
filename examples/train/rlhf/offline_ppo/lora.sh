#!/bin/bash
# ============================================================
# Offline PPO Training Script
# Usage: bash examples/train/rlhf/offline_ppo/lora.sh
# ============================================================

# ======================== 按需修改区 ========================

# 模型
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# 数据集路径（jsonl 文件）
DATASET="/path/to/your/dataset.jsonl"

# 输出目录
OUTPUT_DIR="output/offline_ppo"

# GPU 设置
CUDA_VISIBLE_DEVICES="0"
NPROC_PER_NODE=1

# 训练模式: "lora" 或 "full"
TUNER_TYPE="lora"

# 训练超参
NUM_EPOCHS=1
LEARNING_RATE=1e-6
BATCH_SIZE=2
GRAD_ACCUM=8
MAX_LENGTH=4096

# LoRA 参数（仅 TUNER_TYPE=lora 时生效）
LORA_RANK=8
LORA_ALPHA=32

# 离线 PPO 专有
KL_COEF=0.2
CLIPRANGE=0.1
WHITEN_REWARDS=true
# 标量 reward 列名（若设 REWARD_KEYS 则该列会被加权组合覆盖）
REWARD_KEY="expected_acc_reward"
ANSWER_KEY="answer"
# 组合 reward（可选）：列名与权重逗号分隔，例如 acc + 0.5*llm_acc + llm_score
# REWARD_KEYS="acc,llm_acc,llm_score"
# REWARD_WEIGHTS="1,0.5,1"

# 保存 & 日志
SAVE_STRATEGY="epoch"       # "epoch" 按轮保存, "steps" 按步保存
SAVE_STEPS=200              # 仅 SAVE_STRATEGY=steps 时生效
SAVE_TOTAL_LIMIT=3
LOGGING_STEPS=5
EVAL_RATIO=0.01

# 日志平台: "tensorboard" 或 "wandb"（可同时用: "tensorboard wandb"）
REPORT_TO="tensorboard"
# wandb 设置（仅 REPORT_TO 含 wandb 时生效）
export WANDB_PROJECT="${WANDB_PROJECT:-offline-ppo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"  # "offline" 离线记录, "online" 实时上传

# ======================== 构建命令 ========================

ARGS=(
    --rlhf_type offline_ppo
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
    --weight_decay 0.1
    --save_strategy ${SAVE_STRATEGY}
    --save_steps ${SAVE_STEPS}
    --save_total_limit ${SAVE_TOTAL_LIMIT}
    --save_only_model true
    --logging_steps ${LOGGING_STEPS}
    --split_dataset_ratio ${EVAL_RATIO}
    --eval_strategy ${SAVE_STRATEGY}
    --eval_steps ${SAVE_STEPS}
    --dataloader_num_workers 4
    --offline_ppo_kl_coef ${KL_COEF}
    --offline_ppo_cliprange ${CLIPRANGE}
    --offline_ppo_whiten_rewards ${WHITEN_REWARDS}
    --offline_ppo_reward_key "${REWARD_KEY}"
    --offline_ppo_answer_key "${ANSWER_KEY}"
    --report_to ${REPORT_TO}
)

if [ -n "${REWARD_KEYS:-}" ]; then
    ARGS+=(--offline_ppo_reward_keys "${REWARD_KEYS}")
fi
if [ -n "${REWARD_WEIGHTS:-}" ]; then
    ARGS+=(--offline_ppo_reward_weights "${REWARD_WEIGHTS}")
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
