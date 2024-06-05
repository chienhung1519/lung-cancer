from argparse import ArgumentParser, Namespace
import numpy as np
import jsonlines
from pathlib import Path
from typing import List, Dict
import json

from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
import wandb


TARGETS = ["NonImmu", "CK7", "TTF-1", "CK20", "P40"]

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--ntu_file", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--mapping_file", type=str, default="./src/label_mappings.json")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=7687)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    args = parser.parse_args()
    return args

def load_data(data_path: str) -> List[Dict]:
    with jsonlines.open(data_path) as reader:
        data = [obj for obj in reader]
    return data

def get_label_list(dataset, tag_name):
    label_list = []
    for i in range(len(dataset)):
        label_list.extend(dataset[i][tag_name])
    return list(set(label_list))

def save_jsonl(path: str, data: List[Dict]):
    """Save data to jsonl file."""
    with jsonlines.open(path, "w") as writer:
        writer.write_all(data)

def main():

    args = parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = load_data(args.train_file)
    train_data, val_data = train_test_split(train_data, test_size=args.val_size, random_state=args.seed)
    test_data = load_data(args.test_file)
    ntu_data = load_data(args.ntu_file)

    mappings = json.loads(Path(args.mapping_file).read_text())

    for target in TARGETS:

        datasets = DatasetDict({
            "train": Dataset.from_list(train_data),
            "val": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data),
            "ntu": Dataset.from_list(ntu_data),
        })

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        # Label list
        label_list = list(mappings[target].keys())
        label_to_id = mappings[target]

        b_to_i_label = [] # B-XXX -> I-XXX
        for idx, label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

        # Dataset preprocessing
        text_column_name = "tokens"
        label_column_name = f"{target}_tags"
        label_all_tokens = False

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name], 
                padding=True, 
                truncation=True, 
                max_length=args.max_seq_length,
                is_split_into_words=True, # use when input is tokenized as list of words
            )

            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label_to_id[label[word_idx]])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        if label_all_tokens:
                            label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                        else:
                            label_ids.append(-100)
                    previous_word_idx = word_idx

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels

            return tokenized_inputs
        
        datasets = datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=datasets["train"].column_names,
        )

        # Model
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, num_labels=len(label_list))

        # Metric
        metric = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{args.output_dir}/sft/{target}",
            overwrite_output_dir=True,
            num_train_epochs=args.num_train_epochs,
            learning_rate=2e-5,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            logging_steps=20,
            save_strategy="epoch",
            save_total_limit=1,
            evaluation_strategy="epoch", # evaluation_strategy must equal to save_strategy ["epoch", "steps"]
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer)

        # Callback
        early_sttoping_callback = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)

        # Wandb
        wandb.init(project="lung-cancer", name=f"{args.model_name_or_path}-{target}")

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[early_sttoping_callback],
        )

        # Train
        trainer.train()

        # Stop wandb
        wandb.finish()

        # Inference test data
        predictions = trainer.predict(datasets["test"])
        preds = predictions.predictions.argmax(axis=-1)
        labels = predictions.label_ids

        true_preds = [
            [label_list[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(preds, labels)
        ]

        for example, pred in zip(test_data, true_preds):
            example[f"{target}_pred"] = pred

        # Inference ntu data
        predictions = trainer.predict(datasets["ntu"])
        preds = predictions.predictions.argmax(axis=-1)
        labels = predictions.label_ids

        true_preds = [
            [label_list[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(preds, labels)
        ]

        for example, pred in zip(ntu_data, true_preds):
            example[f"{target}_pred"] = pred

        # Delete model
        del model

    save_jsonl(Path(args.output_dir, "predict", "tmu", "generated_predictions.jsonl"), test_data)
    save_jsonl(Path(args.output_dir, "predict", "ntu", "generated_predictions.jsonl"), ntu_data)

if __name__ == "__main__":
    main()