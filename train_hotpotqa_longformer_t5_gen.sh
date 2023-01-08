#!/bin/bash 

CUDA_VISIBLE_DEVICES=0,1 python run_qa.py \
--model_name_or_path "allenai/longformer-base-4096" \
--dataset_name "hotpotqa_t5_gen" \
--generated_dataset_path "/home/deokhk/research/HotpotQA_longformer/data/t5_generated" \
--do_train \
--do_eval \
--seed 42 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
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
--run_name longformer_hotpotqa_t5_gen \
--output_dir /home/deokhk/research/longformer_trained_models/t5_generated \
--overwrite_output_dir \
--report_to "wandb" 

# GPU 4장으로 학습. 2(per device batch) X 4(grad_accum) X 4 (num_gpu)