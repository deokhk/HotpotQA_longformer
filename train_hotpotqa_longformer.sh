#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_qa.py \
--model_name "allenai/longformer-base-4096" \
--dataset_name "hotpotqa_longformer" \
--do_train \
--do_eval \
--seed 42 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--learning_rate 3e-5 \
--num_train_epochs 3 \
--save_strategy steps \
--warmup_steps 1000 \
--save_steps 1000 \
--logging_steps 50 \
--evaluation_strategy steps \
--eval_steps 1000 \
--weight_decay 1e-2 \
--load_best_model_at_end True \
--save_total_limit 2 \
--fp16 True \
--run_name longformer_hotpotqa_test \
--output_dir /home/deokhk/research/longformer_trained_models/hotpotqa \
--overwrite_output_dir \
--prediction_loss_only True \
--ddp_find_unused_parameters False \
--report_to "wandb" 
