# Offline PPO with pre-collected responses and rewards
# Dataset format (jsonl):
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

nproc_per_node=1

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type offline_ppo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset /path/to/your/dataset.jsonl \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output/offline_ppo \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --offline_ppo_kl_coef 0.05 \
    --offline_ppo_cliprange 0.2 \
    --offline_ppo_whiten_rewards true \
    --offline_ppo_reward_key expected_acc_reward \
    --offline_ppo_answer_key answer \
    --save_only_model true
