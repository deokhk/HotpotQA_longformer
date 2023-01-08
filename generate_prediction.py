# Generate prediction using pretrained models.

import argparse
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)



def main(args):
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForQuestionAnswering.from_pretrained()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", help="Path to the pretrained model checkpoint", default="")
    parser.add_argument("--predicted_file_path", help="Model prediction will be saved here", default="")

    args = parser.parse_args()
    main(args)