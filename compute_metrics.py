from argparse import ArgumentParser
import json
import pandas as pd


def parse_args():
    parser = ArgumentParser(description="Compute metrics for the model")
    parser.add_argument("--predictions_file", type=str, help="Path to the predictions")
    parser.add_argument("--outputs_dir", type=str, help="Path to the output file")
    return parser.parse_args()


def load_from_jsonl(file_name: str) -> list[dict]:
    def load_json_line(line: str, i: int, file_name: str):
        try:
            json_line = json.loads(line)
            return {k: json.loads(v) for k, v in json_line.items()}
        except:
            raise ValueError(f"Error in line {i+1}\n{line} of {file_name}")
    with open(file_name, "r") as f:
        data = [load_json_line(line, i, file_name) for i, line in enumerate(f)]
    return data


def metrics(labels, preds):
    tp, num_labels, num_preds = 0, 0, 0
    for label, pred in zip(labels, preds):
        if label != "unknown" and label == pred: # true positive
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

    # Load predictions
    results = load_from_jsonl(args.predictions_file)
    labels = [result["label"] for result in results]
    predictions = [result["predict"] for result in results]

    # Compute metrics
    targets = list(labels[0].keys())
    classification_report = []
    confusion_matrix = []
    for target in targets:
        labels_target = [label[target] for label in labels]
        predictions_target = [pred[target] for pred in predictions]
        metrics_dict = metrics(labels_target, predictions_target)
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