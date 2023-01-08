#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python run_qa.py \
--model_name "allenai/longformer-base-4096" \
--dataset_name "hotpotqa_dire_simple" \
--do_train \
--do_eval \
--seed 42 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
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
--run_name longformer_hotpotqa_simple_input \
--output_dir /home/deokhk/research/longformer_trained_models/hotpotqa_simple \
--overwrite_output_dir \
--report_to "wandb" 
