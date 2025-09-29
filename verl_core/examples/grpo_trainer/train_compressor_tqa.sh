set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="your api key"

model_path="../../../LLaMA-Factory/saves/qwen2.5_1.5B/full/deepseek_distill_sft/tqa_lr5e5_epoch2_bs128"
data_path="../../data"

qa_train_path=$data_path/tqa/train_sample_all.parquet
qa_test_path=$data_path/tqa/eval_sample_500_v2.parquet

train_files="['$qa_train_path']"
test_files="['$qa_test_path']"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1600 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_tqa' \
    trainer.experiment_name='rollout5_lr1e5_alldata_rw4' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=305 \
    trainer.test_freq=305 \
    trainer.total_epochs=10 \
    trainer.resume_mode=disable \
    custom_reward_function.path="../../customized_rewards_asyn.py" \
    custom_reward_function.name=compute_score_tqa \
    reward_model.reward_manager=apiprime \
    reward_model.launch_reward_fn_async=True $@

    #   trainer.resume_from_path=False \