swift rlhf \
    --rlhf_type offline_ppo \
    --model /mnt/gpfs/xuexiangyuan/model/qwen3-vl-8b-instruct \
    --dataset /path/to/your/data.jsonl \
    --tuner_type full \
    --offline_ppo_kl_coef 0.05 \
    --offline_ppo_cliprange 0.2 \
    --offline_ppo_whiten_rewards true \
    --offline_ppo_reward_key expected_acc_reward \
    --offline_ppo_answer_key answer