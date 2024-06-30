#!/bin/bash

is_causal=True
is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="meta-llama"
model_type="Meta-Llama-3-8B-Instruct"
quantization_type="origin"
peft_type="origin"
loss_type="ordinal_log_loss"
data_max_length=598
target_max_length=2
precision="bf16"
batch_size=128

python main.py mode=train \
    is_causal=$is_causal \
    is_preprocessed=$is_preprocessed \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    quantization_type=$quantization_type \
    peft_type=$peft_type \
    data_max_length=$data_max_length \
    target_max_length=$target_max_length \
    precision=$precision \
    batch_size=$batch_size
