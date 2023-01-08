## SQuAD evaluation script. Modifed slightly for this notebook

from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import os 

from datasets import load_dataset, DatasetDict, Dataset 
from random import Random 

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
      total += 1
      exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
      f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

def load_hotpotqa_for_longformer():
    """
    Load hotpotqa dataset for longformer.
    Here, the context is processed like this:


    We need to return three things in order to match the SQuAD format.
    [1] ["question"] = question 
    [2] ["answers"] = {"text": [answer_text], "answer_start": [answer_start]}
    [3] ["context"] = context 

    And the context looks like this:
    [CLS] question [SEP] <yes> <no> p1_1 p1_2 \n p2_1 p2_2 .. 
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
        mycontext = "<yes> <no> "
        for title, sentence_list in zip(context_titles, context_sentences_list):
            for sentence in sentence_list:
                mycontext += sentence
            mycontext += "\n"
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
        mycontext = "<yes> <no> "
        for title, sentence_list in zip(context_titles, context_sentences_list):
            for sentence in sentence_list:
                mycontext += sentence
            mycontext += "\n"

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


    dataset = DatasetDict({"train": train_dataset, "validation": dev_dataset})
    return dataset

def load_hotpotqa_for_longformer_dire_qa_simple():
    """
    Load hotpotqa dataset for longformer, for testing the amount of disconnected reasoning
    in the given dataset.

    Here, the context is processed like this:


    We need to return three things in order to match the SQuAD format.
    [1] ["question"] = question 
    [2] ["answers"] = {"text": [answer_text], "answer_start": [answer_start]}
    [3] ["context"] = context 

    And the context looks like this:
    [CLS] question [SEP] <yes> <no> p1_1 p1_2 \n p2_1 p2_2 .. 

    One difference is that we only use half of the hotpotqa train dataset.
    This is intendeded.
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
        mycontext = "<yes> <no> "
        for title, sentence_list in zip(context_titles, context_sentences_list):
            for sentence in sentence_list:
                mycontext += sentence
            mycontext += "\n"
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
        mycontext = "<yes> <no> "
        for title, sentence_list in zip(context_titles, context_sentences_list):
            for sentence in sentence_list:
                mycontext += sentence
            mycontext += "\n"

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

    # For QA, we will only use the first half of the train dataset. 
    # For deterministic split, we set random seed before shuffling.

    Random(42).shuffle(train_data_cleaned)
    train_data_cleaned = train_data_cleaned[0:len(train_data_cleaned)//2]


    train_dataset  = Dataset.from_list(train_data_cleaned)
    dev_dataset = Dataset.from_list(validation_data_cleaned)


    dataset = DatasetDict({"train": train_dataset, "validation": dev_dataset})
    return dataset


def load_hotpotqa_for_longformer_dire_t5(t5_generated_dataset_dir:str):
    """
    Load hotpotqa dataset for longformer, where the dataset is generated by t5 model.
    Dataset statistics
    Train: 45212
    Dev:7404

    """
    print(f"Loaded generated dataset from {t5_generated_dataset_dir}")
    t5_gen_train_path = os.path.join(t5_generated_dataset_dir, "t5_train_generated.json")
    t5_gen_valid_path = os.path.join(t5_generated_dataset_dir, "t5_valid_generated.json")

    with open(t5_gen_train_path, 'r') as f: 
        gen_train = json.load(f)

    with open(t5_gen_valid_path, 'r') as f: 
        gen_valid = json.load(f)

    assert len(gen_train) < 50000, "Generated training dataset must be generated from half of the original training dataset in advance!"
    print("Loading completed!")
    train_data_cleaned = []

    for datapoint in gen_train: 
        cleaned_qapair = dict()
        mycontext = ""
        context_titles = datapoint["context"]["title"]
        context_sentences_list = datapoint["context"]["sentences"]
        question = datapoint["question"]
        mycontext = "<yes> <no> "
        for title, sentence_list in zip(context_titles, context_sentences_list):
            for sentence in sentence_list:
                mycontext += sentence
            mycontext += "\n"
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

    validation_data_cleaned = []

    for datapoint in gen_valid: 
        cleaned_qapair = dict()
        mycontext = ""
        context_titles = datapoint["context"]["title"]
        context_sentences_list = datapoint["context"]["sentences"]
        question = datapoint["question"]
        mycontext = "<yes> <no> "
        for title, sentence_list in zip(context_titles, context_sentences_list):
            for sentence in sentence_list:
                mycontext += sentence
            mycontext += "\n"

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

    # Here, the split of the train dataset is not performed
    # Since the train dataset is already splitted.


    train_dataset  = Dataset.from_list(train_data_cleaned)
    dev_dataset = Dataset.from_list(validation_data_cleaned)


    dataset = DatasetDict({"train": train_dataset, "validation": dev_dataset})
    return dataset


"""
EXPERIMENT: Measure the amount of disconnected reasoning in 
Original HotpotQA dataset
"""

def dataset_builder_from_json_loaded_train_and_dev(train, dev):
    train_data_cleaned = []

    for datapoint in train: 
        cleaned_qapair = dict()
        mycontext = ""
        contexts = datapoint["context"]
        question = datapoint["question"]
        mycontext = "<yes> <no> "
        for context in contexts:
            context_title = context[0]
            context_sentence_list = context[1]
            for sentence in context_sentence_list:
                mycontext += sentence
            mycontext += "\n"
        answer_text = datapoint["answer"]
        if answer_text in mycontext:
            answer_start_idx = mycontext.index(answer_text)
        else:
            answer_start_idx = -1
            
        cleaned_qapair["question"] = question 
        cleaned_qapair["answers"] = {"text": [answer_text], "answer_start": [answer_start_idx]}
        cleaned_qapair["context"] = mycontext 
        cleaned_qapair["id"] = datapoint["_id"]
        train_data_cleaned.append(cleaned_qapair)

    validation_data_cleaned = []

    for datapoint in dev: 
        cleaned_qapair = dict()
        mycontext = ""
        contexts = datapoint["context"]
        question = datapoint["question"]
        mycontext = "<yes> <no> "
        for context in contexts:
            context_title = context[0]
            context_sentence_list = context[1]
            for sentence in context_sentence_list:
                mycontext += sentence
            mycontext += "\n"

        answer_text = datapoint["answer"]
        if answer_text in mycontext:
            answer_start_idx = mycontext.index(answer_text)
        else:
            answer_start_idx = -1 
                    
        cleaned_qapair["question"] = question 
        cleaned_qapair["answers"] = {"text": [answer_text], "answer_start": [answer_start_idx]}
        cleaned_qapair["context"] = mycontext 
        cleaned_qapair["id"] = datapoint["_id"]
        validation_data_cleaned.append(cleaned_qapair)

    # Here, the split of the train dataset is not performed
    # Since the train dataset is already splitted.


    train_dataset  = Dataset.from_list(train_data_cleaned)
    dev_dataset = Dataset.from_list(validation_data_cleaned)


    dataset = DatasetDict({"train": train_dataset, "validation": dev_dataset})
    return dataset

"""
ID 중복되는 경우 해결.
id set에 이미 존재하는 ID가 들어오면
ID = ID_01로 ID 바꿔쳐서 저장 
"""
def dataset_builder_from_json_loaded_train_and_dev_probe(train, dev):
    train_data_cleaned = []

    for datapoint in train: 
        cleaned_qapair = dict()
        mycontext = ""
        contexts = datapoint["context"]
        question = datapoint["question"]
        mycontext = "<yes> <no> "
        for context in contexts:
            context_title = context[0]
            context_sentence_list = context[1]
            for sentence in context_sentence_list:
                mycontext += sentence
            mycontext += "\n"
        answer_text = datapoint["answer"]
        if answer_text in mycontext:
            answer_start_idx = mycontext.index(answer_text)
        else:
            answer_start_idx = -1
            
        cleaned_qapair["question"] = question 
        cleaned_qapair["answers"] = {"text": [answer_text], "answer_start": [answer_start_idx]}
        cleaned_qapair["context"] = mycontext 
        cleaned_qapair["id"] = datapoint["_id"]
        train_data_cleaned.append(cleaned_qapair)

    validation_data_cleaned = []

    dev_id_set = set()
    for datapoint in dev: 
        cleaned_qapair = dict()
        mycontext = ""
        contexts = datapoint["context"]
        question = datapoint["question"]
        mycontext = "<yes> <no> "
        for context in contexts:
            context_title = context[0]
            context_sentence_list = context[1]
            for sentence in context_sentence_list:
                mycontext += sentence
            mycontext += "\n"

        answer_text = datapoint["answer"]
        if answer_text in mycontext:
            answer_start_idx = mycontext.index(answer_text)
        else:
            answer_start_idx = -1 
                    
        cleaned_qapair["question"] = question 
        cleaned_qapair["answers"] = {"text": [answer_text], "answer_start": [answer_start_idx]}
        cleaned_qapair["context"] = mycontext 
        devid = datapoint["_id"]
        if devid in dev_id_set:
            devid = devid+"_01"
        else:
            dev_id_set.add(devid)
        cleaned_qapair["id"] = devid
        validation_data_cleaned.append(cleaned_qapair)

    # Here, the split of the train dataset is not performed
    # Since the train dataset is already splitted.


    train_dataset  = Dataset.from_list(train_data_cleaned)
    dev_dataset = Dataset.from_list(validation_data_cleaned)


    dataset = DatasetDict({"train": train_dataset, "validation": dev_dataset})
    return dataset


def load_hotpotqa_dire_filtered_original(processed_dataset_dir:str):
    # By default,
    # processed_dataset_dir -> /home/deokhk/research/dire/data/processed
    # original dataset (HotpotQA)

    train_path = os.path.join(processed_dataset_dir, "original_hotpotqa_train.json")
    dev_path = os.path.join(processed_dataset_dir, "original_hotpotqa_dev.json")

    with open(train_path, 'r') as f:
        train = json.load(f)
    
    with open(dev_path, 'r') as f:
        dev = json.load(f)
    
    dataset = dataset_builder_from_json_loaded_train_and_dev(train, dev)
    return dataset

def load_probe_hotpotqa_original(processed_dataset_dir:str):
    # By default,
    # processed_dataset_dir -> /home/deokhk/research/dire/data/processed
    # HotpotQA에 probe apply 

    train_path = os.path.join(processed_dataset_dir, "probe_of_original_hotpotqa_train.json")
    dev_path = os.path.join(processed_dataset_dir, "probe_of_original_hotpotqa_dev.json")

    with open(train_path, 'r') as f:
        train = json.load(f)
    
    with open(dev_path, 'r') as f:
        dev = json.load(f)
    
    dataset = dataset_builder_from_json_loaded_train_and_dev_probe(train, dev)
    return dataset
