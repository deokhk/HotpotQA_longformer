import torch
import datasets
from transformers import LongformerTokenizerFast
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb 

from datasets import load_dataset, Dataset, DatasetDict

from transformers import LongformerForQuestionAnswering, LongformerTokenizerFast, EvalPrediction
from transformers import (
    HfArgumentParser,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

os.environ["WANDB_PROJECT"] = "DIRE"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default="hotpotqa", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_len: Optional[int] = field(
        default=4096,
        metadata={"help": "Max input length for the source text"},
    )


tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', additional_special_tokens=['[q]', '[/q]', '<t>', '</t>', '[s]'])

def get_correct_alignement(context, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text'][0]
    start_idx = answer['answer_start'][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       # When the gold label position is good
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1   # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2   # When the gold label is off by two character
    else:
        raise ValueError()

special_token_ids_list = tokenizer('[q][/q]<t></t>[s]<s></s>')["input_ids"]
special_token_ids_list = list(set(special_token_ids_list))

def func_global_att(x):
    if x in special_token_ids_list:
        return 1 
    else:
        return 0


# Tokenize our training dataset
def convert_to_features(example):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer(example['context'], max_length=4096, padding="max_length")
    input_ids = encodings["input_ids"]
    global_attention_mask = list(map(func_global_att, input_ids))
    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
    # this will give us the position of answer span in the context text
    start_idx, end_idx = get_correct_alignement(example['context'], example['answers'])
    
    context_encodings = tokenizer.encode_plus(example['context'])

    start_positions = context_encodings.char_to_token(start_idx)
    end_positions = context_encodings.char_to_token(end_idx-1)
    if end_positions > 4096:
      start_positions, end_positions = 0, 0

    encodings.update({'start_positions': start_positions,
                      'end_positions': end_positions,
                      'attention_mask': encodings['attention_mask'],
                      'global_attention_mask': global_attention_mask})

    return encodings

def load_hotpotqa_for_longformer():
    """
    Load hotpotqa dataset for longformer.
    Here, the context is processed like this:


    We need to return three things in order to match the SQuAD format.
    [1] ["question"] = question 
    [2] ["answers"] = {"text": [answer_text], "answer_start": [answer_start]}
    [3] ["context"] = context 

    And the context looks like this:
    [CLS] [q] question [/q] <yes> <no> <t> title_1 </t> sent_1,1 [s] sent_1,2 [s] .. <t> 
    title_2 </t> sent_2,1 [s] sent_2,2 [s] .. "

    Here, [q], [/q], <t>, </t>, [s], <yes>, <no> is all special tokens.
    """
    hotpotqa_distractor = load_dataset("hotpot_qa","distractor")
    hotpotqa_train = hotpotqa_distractor["train"]
    train_data_cleaned = []

    for datapoint in hotpotqa_train: 
        cleaned_qapair = dict()
        mycontext = ""
        context_titles = datapoint["context"]["title"]
        context_sentences_list = datapoint["context"]["sentences"]
        question = datapoint["question"]
        mycontext = f"[q] {question} [/q] <yes> <no> "
        for title, sentence_list in zip(context_titles, context_sentences_list):
            mycontext += (f"<t> {title} </t> ")
            for sentence in sentence_list:
                mycontext += (f"{sentence} [s] ")
        answer_text = datapoint["answer"]
        if answer_text in mycontext:
            answer_start_idx = mycontext.index(answer_text)
        else:
            print("Skipped this qapair. The answer does not exist as a span in the context.")
            continue 
            
        cleaned_qapair["question"] = question 
        cleaned_qapair["answers"] = {"text": [answer_text], "answer_start": [answer_start_idx]}
        cleaned_qapair["context"] = mycontext 
        cleaned_qapair["id"] = datapoint["id"]
        train_data_cleaned.append(cleaned_qapair)

    hotpotqa_validation = hotpotqa_distractor["validation"]
    validation_data_cleaned = []

    for datapoint in hotpotqa_validation: 
        cleaned_qapair = dict()
        mycontext = ""
        context_titles = datapoint["context"]["title"]
        context_sentences_list = datapoint["context"]["sentences"]
        question = datapoint["question"]
        mycontext = f"[q] {question} [/q] <yes> <no> "
        for title, sentence_list in zip(context_titles, context_sentences_list):
            mycontext += (f"<t> {title} </t> ")
            for sentence in sentence_list:
                mycontext += (f"{sentence} [s] ")
        answer_text = datapoint["answer"]
        if answer_text in mycontext:
            answer_start_idx = mycontext.index(answer_text)
        else:
            print("Skipped this qapair. The answer does not exist as a span in the context.")
            continue 

        cleaned_qapair["question"] = question 
        cleaned_qapair["answers"] = {"text": [answer_text], "answer_start": [answer_start_idx]}
        cleaned_qapair["context"] = mycontext 
        cleaned_qapair["id"] = datapoint["id"]
        validation_data_cleaned.append(cleaned_qapair)


    train_dataset  = Dataset.from_list(train_data_cleaned)
    dev_dataset = Dataset.from_list(validation_data_cleaned)

    return (train_dataset, dev_dataset)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    wandb.login()


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file, 
    # make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    num_gpu_available = torch.cuda.device_count()
    num_workers = int(4 * num_gpu_available)

    training_args.dataloader_num_workers=num_workers

    training_args.run_name = f"longformer_{data_args.dataset_name}_b{num_gpu_available*training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps}"
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = LongformerTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        additional_special_tokens=['[q]', '[/q]', '<t>', '</t>', '[s]']
    )

    model = LongformerForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))
    # Get datasets
    print("Loading datasets..")
    if data_args.dataset_name == "hotpotqa_longformer":
        train_dataset, valid_dataset = load_hotpotqa_for_longformer()

    train_dataset = train_dataset.map(convert_to_features)
    valid_dataset = valid_dataset.map(convert_to_features, load_from_cache_file=False)


    # set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions', 'global_attention_mask']
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)
    print("Loading done!")


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DefaultDataCollator(),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if training_args.local_rank == 0:
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results

if __name__ == "__main__":
    main()