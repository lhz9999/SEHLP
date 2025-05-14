#!/bin/bash
CUDA_VISIBLE_DEVICES=0
PORT=7012

# 设置多进程启动方法为spawn
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 启动服务
/root/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/PLMs/Qwen2_5_14B_Instruct \
    --port $PORT \
    --trust-remote-code \
    --served-model-name "model" \
    --tensor-parallel-size 1 \
    --disable-log-requests \
    --disable-log-stats \
    --gpu-memory-utilization 0.95 \
    --max-model-len 2048