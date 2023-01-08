#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python run_qa.py \
--model_name_or_path "/home/deokhk/research/longformer_trained_models/hotpotqa_simple" \
--dataset_name "hotpotqa_dire_filtered_original" \
--generated_dataset_path "/home/deokhk/research/dire/data/processed" \
--do_eval \
--seed 42 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 8 \
--learning_rate 5e-5 \
--num_train_epochs 5 \
--warmup_steps 1000 \
--save_strategy steps \
--save_steps 1000 \
--logging_steps 50 \
--max_seq_length 4096 \
--evaluation_strategy steps \
--eval_steps 1000 \
--metric_for_best_model="f1" \
--load_best_model_at_end True \
--save_total_limit 2 \
--fp16 True \
--run_name hotpotqa_dire_filtered_original_eval_only \
--output_dir /home/deokhk/research/longformer_trained_models/hotpotqa_dire_filtered_original_eval_only \
--report_to "wandb" 
