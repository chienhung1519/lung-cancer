from argparse import ArgumentParser
import json
import pandas as pd
import re
from pathlib import Path


TARGETS = [
    "ALK", "EGFR_18", "EGFR_19", "EGFR_20", "EGFR_21", "Htype", "LV_invasion", "PDL1_immune", 
    "ROS1", "diagnosis", "operation", "organ", "CK7", "TTF-1", "CK20", "P40"
]


def parse_args():
    parser = ArgumentParser(description="Compute metrics for the model")
    parser.add_argument("--predictions_file", type=str, help="Path to the predictions file")
    parser.add_argument("--outputs_dir", type=str, help="Path to the output file")
    parser.add_argument("--model_type", type=str, choices=["llama", "t5", "bert"])
    return parser.parse_args()


def load_from_jsonl(file_name: str) -> list[dict]:
    def load_json_line(line: str, i: int, file_name: str):
        try:
            return json.loads(line)
        except:
            raise ValueError(f"Error in line {i+1}\n{line} of {file_name}")
    with open(file_name, "r") as f:
        data = [load_json_line(line, i, file_name) for i, line in enumerate(f)]
    return data


def extract_llama_labels_and_predictions(results: list[dict]) -> tuple[list[dict], list[dict]]:
    labels = [json.loads(result["label"]) for result in results]
    predictions = []
    for result in results:
        try:
            predictions.append(json.loads(result["predict"]))
        except:
            predictions.append({t: "unknown" for t in TARGETS})
    return labels, predictions


def extract_t5_labels_and_predictions(results: list[dict]) -> tuple[list[dict], list[dict]]:
    labels = [json.loads(result["target_text"]) for result in results]
    predictions = [json.loads("{"+result["prediction"]+"}") for result in results]
    return labels, predictions


def tags_to_json(tags: list[str], tokens: list[str]) -> dict:
    json_tags = {}
    cur_tag = None
    for tag, token in zip(tags, tokens):
        if tag.startswith("B-"):
            tag_name = tag[2:]
            json_tags[tag_name] = [token]
            cur_tag = tag_name
        elif tag.startswith("I-"):
            if cur_tag is not None:
                json_tags[cur_tag].append(token)
            else:
                tag_name = tag[2:]
                json_tags[tag_name] = [token]
                cur_tag = tag_name
        else:
            cur_tag = None
    json_tags = {k: " ".join(v) for k, v in json_tags.items()}
    return json_tags


def extract_bert_labels_and_predictions(results: list[dict]) -> tuple[list[dict], list[dict]]:
    labels, predictions = [], []
    for result in results:
        e_labels, e_predictions = {}, {}
        for target in ["NonImmu", "CK7", "TTF-1", "CK20", "P40"]:
            e_labels.update(tags_to_json(result[f"{target}_tags"], result["tokens"]))
            e_predictions.update(tags_to_json(result[f"{target}_pred"], result["tokens"]))
        labels.append(e_labels)
        predictions.append(e_predictions)
    return labels, predictions


def extract_labels_and_predictions(file_name: str, model_type: str) -> tuple[list[dict], list[dict]]:
    results = load_from_jsonl(file_name)
    if model_type == "llama":
        return extract_llama_labels_and_predictions(results)
    elif model_type == "t5":
        return extract_t5_labels_and_predictions(results)
    elif model_type == "bert":
        return extract_bert_labels_and_predictions(results)
    else:
        raise ValueError("Invalid model type")


def fill_in_unknown(example: dict) -> dict:
    for target in TARGETS:
        if target not in example:
            example[target] = "unknown"
    return example


def clear_text(text: str) -> str:
    clean_text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    clean_text = re.sub(r"\s+", " ", clean_text)
    if text in ["(+)", "(-)", "(+", "(-", "+)", "-)"]:
        clean_text = clean_text.replace("(", "").replace(")", "")
    return clean_text


def metrics(labels, preds, method="hard_match"):
    tp, num_labels, num_preds = 0, 0, 0
    for label, pred in zip(labels, preds):
        if label != "unknown":
            if method == "hard_match" and label == pred: # true positive
                tp += 1
            elif method == "soft_match" and pred in label: # true positive
                tp += 1
        num_labels += 1 if label != "unknown" else 0
        num_preds += 1 if pred != "unknown" else 0
    precision = tp / num_preds if num_preds != 0 else 0
    recall = tp / num_labels if num_labels != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "num_labels": num_labels,
        "num_preds": num_preds
    }

def main():
    args = parse_args()
    Path(args.outputs_dir).mkdir(parents=True, exist_ok=True)

    # Load predictions
    labels, predictions = extract_labels_and_predictions(args.predictions_file, args.model_type)

    # Clear text
    labels = [{k: clear_text(v) for k, v in label.items()} for label in labels]
    predictions = [{k: clear_text(v) for k, v in prediction.items()} for prediction in predictions]

    # Fill in unknown
    labels = [fill_in_unknown(label) for label in labels]
    predictions = [fill_in_unknown(prediction) for prediction in predictions]

    # Compute metrics
    classification_report = []
    confusion_matrix = []
    for target in TARGETS:
        labels_target = [label[target] for label in labels]
        predictions_target = [pred[target] for pred in predictions]
        method = "soft_match" if target == "Htype" else "hard_match"
        metrics_dict = metrics(labels_target, predictions_target, method)
        classification_report.append(
            [target, metrics_dict["precision"], metrics_dict["recall"], metrics_dict["f1"]]
        )
        confusion_matrix.append(
            [target, metrics_dict["tp"], metrics_dict["num_labels"], metrics_dict["num_preds"]]
        )
        
    # Micro-average
    micro_precision = sum([m[1] for m in confusion_matrix]) / sum([m[3] for m in confusion_matrix])
    micro_recall = sum([m[1] for m in confusion_matrix]) / sum([m[2] for m in confusion_matrix])
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) != 0 else 0

    # Macro-average
    macro_precision = sum([r[1] for r in classification_report]) / len(classification_report)
    macro_recall = sum([r[2] for r in classification_report]) / len(classification_report)
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) != 0 else 0

    # Weighted-average
    target_num_labels = [m[2] for m in confusion_matrix]
    target_weights = [num_labels / sum(target_num_labels) for num_labels in target_num_labels]
    weighted_precision = sum([r[1] * w for r, w in zip(classification_report, target_weights)])
    weighted_recall = sum([r[2] * w for r, w in zip(classification_report, target_weights)])
    weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) != 0 else 0

    classification_report.append(["micro-average", micro_precision, micro_recall, micro_f1])
    classification_report.append(["macro-average", macro_precision, macro_recall, macro_f1])
    classification_report.append(["weighted-average", weighted_precision, weighted_recall, weighted_f1])

    # Save metrics
    pd.DataFrame(
        classification_report, columns=["target", "precision", "recall", "f1"]
    ).to_excel(f"{args.outputs_dir}/classification_report.xlsx", index=False)
    pd.DataFrame(
        confusion_matrix, columns=["target", "tp", "num_labels", "num_preds"]
    ).to_excel(f"{args.outputs_dir}/confusion_metrics.xlsx", index=False)


if __name__ == "__main__":
    main()