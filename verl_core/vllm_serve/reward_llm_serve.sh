
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_NAME="/path/to/the/downstream_LLM"
vllm serve $MODEL_NAME --served-model-name qwen2.5_14b_instruct --port 8000 --tensor-parallel-size 8


