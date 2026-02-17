# The code of paper "Less Is More: Elevating RAG via Performance-Driven Context Compression"

## Distillation for Warm-Start
The distillation stage is based on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory).

Please refer to their documentation for environment setup.

After completing the setup and placing the downloaded data into the data directory, execute the following commands to begin distillation training:

```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_full/qwen2_5_full_sft_nq_lr5e5_epoch2_bs128.yaml
```

## Performance-Driven RL Training

The training code is based on [verl](https://github.com/volcengine/verl).

1.  **Deploy the QA LLM and Start the vLLM Service:**
    First, run `reward_llm_serve.sh` to deploy the QA LLM (Large Language Model). This model is responsible for generating answers, which are subsequently used for reward calculation.

2.  **Configure the Reward Service:**
    Please update the `verl_core/reward.yaml` configuration file with your specific vLLM deployment IP address and the model name.

3.  **Update the Configuration Path:**
    Then, navigate to line 105 in the file `verl_core/verl/workers/reward_manager/api_prime.py`, where you will find the following line of code:
    ```python
    config_path = "verl_core/reward.yaml"
    ```
    Ensure this `config_path` variable points to the correct location of your modified `reward.yaml` file.
```bash
    cd verl_core
    sh examples/grpo_trainer/run_qwen2.5-1.5b_compress_nq.sh
```
