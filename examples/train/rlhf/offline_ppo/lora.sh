#!/bin/bash
# ============================================================
# Offline PPO Training Script
# Usage: bash examples/train/rlhf/offline_ppo/lora.sh
# ============================================================
#
# Dataset format (jsonl, one JSON per line):
# {
#   "messages": [
#     {"role": "system", "content": "..."},
#     {"role": "user", "content": "..."}
#   ],
#   "images": ["data:image/png;base64,..."],
#   "answer": "{\"action\":\"delegate_task\", ...}",
#   "expected_acc_reward": 0.25,
#   "task_id": "SGI_Reasoning_0003"
# }

# ======================== 按需修改区 ========================

# 模型
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# 数据集路径（jsonl 文件）
DATASET="/path/to/your/dataset.jsonl"

# 输出目录
OUTPUT_DIR="output/offline_ppo_lora"

# GPU 设置
CUDA_VISIBLE_DEVICES="0"
NPROC_PER_NODE=1          # 多卡改成对应数量，如 4

# 训练超参
NUM_EPOCHS=3
LEARNING_RATE=5e-6
BATCH_SIZE=2               # 每卡 batch size
GRAD_ACCUM=8               # 梯度累积步数，等效 batch = BATCH_SIZE * GRAD_ACCUM * NPROC
MAX_LENGTH=4096

# LoRA 参数（去掉下面三行 + TUNER_TYPE 改为 full 即全量微调）
TUNER_TYPE="lora"
LORA_RANK=8
LORA_ALPHA=32

# 离线 PPO 专有
KL_COEF=0.05               # KL 惩罚系数
CLIPRANGE=0.2              # PPO 裁剪范围
WHITEN_REWARDS=true         # 是否标准化 reward
REWARD_KEY="expected_acc_reward"  # 数据中 reward 字段名
ANSWER_KEY="answer"               # 数据中 response 字段名

# 保存 & 日志
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=3
LOGGING_STEPS=5
EVAL_RATIO=0.01            # 从训练集切出多少比例做验证

# ======================== 启动训练 ========================

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift rlhf \
    --rlhf_type offline_ppo \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --tuner_type "${TUNER_TYPE}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --gradient_checkpointing true \
    --max_length ${MAX_LENGTH} \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --save_only_model true \
    --logging_steps ${LOGGING_STEPS} \
    --split_dataset_ratio ${EVAL_RATIO} \
    --eval_steps ${SAVE_STEPS} \
    --dataloader_num_workers 4 \
    --offline_ppo_kl_coef ${KL_COEF} \
    --offline_ppo_cliprange ${CLIPRANGE} \
    --offline_ppo_whiten_rewards ${WHITEN_REWARDS} \
    --offline_ppo_reward_key "${REWARD_KEY}" \
    --offline_ppo_answer_key "${ANSWER_KEY}" \
    --report_to tensorboard
