#!/bin/bash

path="src/preprocessing"
upload_user="meta-llama"
model_type="Meta-Llama-3-8B-Instruct"
dataset_modes="train test"

python $path/scale_target_column.py
python $path/add_empty_column.py
python $path/make_corpus.py
python $path/train_tokenizer.py

python $path/merge_tokenizer.py upload_user=$upload_user model_type=$model_type
python $path/merge_model.py upload_user=$upload_user model_type=$model_type

for dataset_mode in $dataset_modes
do
    python $path/preprocess_dataset.py mode=$dataset_mode upload_user=$upload_user model_type=$model_type
done
