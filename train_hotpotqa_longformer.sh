#!/bin/bash

python run_qa.py \
--model_name "" \
--dataset_name "hotpotqa" \
--do_train \
--do_eval \
--do_predict \
--seed 42 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--learning_rate 3e-5 \
--num_train_epochs 10 \
--max_seq_length 4096 \
--doc_stride 3072 \
--save_strategy steps \
--save_steps 50 \
--logging_steps 10 \
--evaluation_strategy steps \
--eval_steps 50 \
--metric_for_best_model eval_exact_match \
--load_best_model_at_end True \
--save_total_limit 2 \
--fp16 True \
--run_name rt_gov_b_05 \
--output_dir /home1/deokhk_1/project/AGC_trained_model_checkpoint/reader/synthetic_pretraining/rt_gov_b_05 \
--overwrite_output_dir
