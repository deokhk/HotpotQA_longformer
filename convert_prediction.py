"""
Convert the prediction into format that can be processed by DIRE
"""
import argparse 
import json 
import os 

def main(args):
    dev_path=args.original_dev_prediction_path
    probe_dev_path=args.probe_dev_prediction_path

    with open(dev_path, 'r') as f:
        dev_prediction = json.load(f)
    with open(probe_dev_path, 'r') as f:
        probe_dev_prediction = json.load(f)
    
    # Convert dev prediction
    converted_datapoint = []
    for id, pred_list in dev_prediction.items():
        converted_dict = dict()
        converted_dict["question_id"] = id 
        best_answer = pred_list[0]
        answer_text = best_answer["text"]
        answer_confidence = best_answer["probability"]
        converted_dict["answer"] = answer_text
        converted_dict["answer_confidence"] = answer_confidence
        converted_datapoint.append(converted_dict)

    with open(args.converted_dev_save_path, encoding="utf-8", mode="w") as f:
        for i in converted_datapoint: f.write(json.dumps(i) + "\n")

    # Convert probe dev prediction
    converted_datapoint = []
    for id, pred_list in probe_dev_prediction.items():
        if id.endswith("_01"):
            id = id.split("_")[0]
        converted_dict = dict()
        converted_dict["question_id"] = id 
        best_answer = pred_list[0]
        answer_text = best_answer["text"]
        answer_confidence = best_answer["probability"]
        converted_dict["answer"] = answer_text
        converted_dict["answer_confidence"] = answer_confidence
        converted_datapoint.append(converted_dict)

    with open(args.converted_probe_dev_save_path, encoding="utf-8", mode="w") as f:
        for i in converted_datapoint: f.write(json.dumps(i) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dev_prediction_path", default="/home/deokhk/research/longformer_trained_models/hotpotqa_simple_eval_only/eval_nbest_predictions.json")
    parser.add_argument("--probe_dev_prediction_path", default="/home/deokhk/research/longformer_trained_models/probe_hotpotqa_original_eval_only/eval_nbest_predictions.json")
    parser.add_argument("--converted_dev_save_path", default="./predictions/converted_dev.jsonl")
    parser.add_argument("--converted_probe_dev_save_path", default="./predictions/converted_probe_dev.jsonl")

    args = parser.parse_args()
    main(args)