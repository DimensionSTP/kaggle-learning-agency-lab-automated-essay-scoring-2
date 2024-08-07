#!/bin/bash

path="src/preprocessing"
upload_users=("meta-llama" "meta-llama" "KevSun" "microsoft")
model_types=("Meta-Llama-3-8B-Instruct" "Meta-Llama-3-8B" "IELTS_essay_scoring" "deberta-v3-large")
length=${#model_types[@]}
dataset_modes="train test"

for ((i=0; i<$length; i++))
do
    upload_user=${upload_users[$i]}
    model_type=${model_types[$i]}
    python $path/merge_tokenizer.py upload_user=$upload_user model_type=$model_type
    python $path/merge_model.py upload_user=$upload_user model_type=$model_type
    for dataset_mode in $dataset_modes
    do
        python $path/preprocess_dataset.py mode=$dataset_mode upload_user=$upload_user model_type=$model_type
    done
done
