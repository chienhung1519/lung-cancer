from argparse import ArgumentParser, Namespace
import pandas as pd
import time
import jsonlines
from typing import List, Dict

from sklearn.model_selection import train_test_split
from simpletransformers.t5 import T5Model, T5Args
from transformers import AutoTokenizer
import transformers
from transformers import AddedToken

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--ntu_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--from_flax", action="store_true")
    parser.add_argument("--seed", type=int, default=7687)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    args = parser.parse_args()
    return args

def load_data(data_path: str) -> List[Dict]:
    with jsonlines.open(data_path) as reader:
        data = [obj for obj in reader]
    return data

def dict_to_dataframe(data):
    data = [["lung cancer report extraction", example["text"], example["label"], example["report_id"], example["sentence_id"]] for example in data]
    return pd.DataFrame(data, columns=["prefix", "input_text", "target_text", "report_id", "sentence_id"])

def main():
    start = time.time()
    args = parse_args()

    args.model_name_or_path = args.output_dir
    args.from_flax = False

    # Load data
    train_data = load_data(args.train_file)
    test_data = load_data(args.test_file)
    ntu_data = load_data(args.ntu_file)

    # Convert to dataframe
    train_data = dict_to_dataframe(train_data)
    test_data = dict_to_dataframe(test_data)
    ntu_data = dict_to_dataframe(ntu_data)
    test_to_predict = [f"{row.prefix}: {row.input_text}" for row in test_data.itertuples()]
    ntu_to_predict = [f"{row.prefix}: {row.input_text}" for row in ntu_data.itertuples()]

    # Configure the model
    model_args = T5Args()

    model_args.num_train_epochs = args.num_train_epochs
    model_args.train_batch_size = args.batch_size
    model_args.eval_batch_size = args.batch_size
    model_args.max_seq_length = args.max_seq_length
    model_args.overwrite_output_dir = True
    model_args.n_gpu = args.n_gpu
    model_args.manual_seed = args.seed
    model_args.max_length = 512
    model_args.gradient_accumulation_steps = args.gradient_accumulation_steps

    model_args.use_multiprocessing = False
    model_args.fp16 = False
    model_args.use_multiprocessed_decoding = False

    model_args.output_dir = args.output_dir
    model_args.best_model_dir = f"{args.output_dir}/best_model"
    model_args.save_best_model = True
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.save_optimizer_and_scheduler = False
    model_args.save_steps = -1

    # model_args.report_to="none"
    model_args.tensorboard_dir = "/nfs/nas-7.1/chchen/lung-cancer/runs"
    model_args.no_cache = True
    model_args.wandb_project = "lung-cancer"
    model_args.wandb_kwargs = {"name": args.model_name_or_path}

    # Initialize the tokenizer
    if "fastchat" in args.model_name_or_path:
        DEFAULT_PAD_TOKEN = "[PAD]"
        tokenizer = transformers.T5Tokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.max_seq_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
        for new_token in ["<", "{", "\n", "}", "`", " ", "\\", "^", "\t"]:
            tokenizer.add_tokens(AddedToken(new_token, normalized=False))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


    # Initialize the model
    model = T5Model(
        "t5", args.model_name_or_path, tokenizer=tokenizer, from_flax=args.from_flax, args=model_args
    )

    # Train the model
    # model.train_model(train_data, use_cuda=True)

    # Make predictions with the model
    test_preds = model.predict(test_to_predict)
    test_data["prediction"] = test_preds
    test_data.to_json(f"{args.output_dir}/test_outputs.jsonl", force_ascii=False, orient='records', lines=True)

    ntu_preds = model.predict(ntu_to_predict)
    ntu_data["prediction"] = ntu_preds
    ntu_data.to_json(f"{args.output_dir}/ntu_outputs.jsonl", force_ascii=False, orient='records', lines=True)

    end = time.time()
    print(f"{end-start} seconds.")

if __name__ == "__main__":
    main()