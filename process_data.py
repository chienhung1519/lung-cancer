from argparse import ArgumentParser, Namespace
from typing import Dict, List
import jsonlines
import pandas as pd
from tqdm.auto import tqdm
import json
from pathlib import Path
import stanza
from sklearn.model_selection import train_test_split


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--src_dir", type=str, default="src")
    parser.add_argument("--text_name", type=str, default="text")
    parser.add_argument("--label_name", type=str, default="label")
    parser.add_argument("--split_train_test", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7687)
    args = parser.parse_args()
    return args


def get_targets(resource_dir) -> Dict[str, List[str]]:
    """Get the targets from the resource."""
    return json.loads(Path(resource_dir, "targets.json").read_text())


def get_reference_prefix() -> str:
    """Get the reference prefix."""
    return "\nRef"


def load_data(data_path: str) -> List[Dict]:
    """Load data from jsonl file."""
    with jsonlines.open(data_path) as reader:
        data = [obj for obj in reader]
    return data


def collect_text_label(example: Dict, text_name, label_name) -> Dict:
    """Collect text and label from the example."""
    return {"id": example["id"], "text": example[text_name], "label": example[label_name]}


def filter_reference(example: Dict) -> Dict:
    """Delete the reference part in the text and label."""
    ref_index = example["text"].rfind(get_reference_prefix())
    text = example["text"] if ref_index == -1 else example["text"][:ref_index]
    label = example["label"] if ref_index == -1 else [label for label in example["label"] if label[0] < ref_index]
    return {"id": example["id"], "text": text, "label": label}


def clean_head_space(text: str, index: int) -> int:
    """Delete the space at the head of the label."""
    if text[index] == " ":
        index += 1
    return index


def clean_tail_space(text: str, index: int) -> int:
    """Delete the space at the tail of the label."""
    if text[index-1] == " ":
        index -= 1
    return index


def check_duplicate(prev_labels: List[List], label: List) -> bool:
    """Check the label is not in the middle of a previous label."""
    for prev_label in prev_labels:
        if prev_label[2] == label[2]:
            if label[0] <= prev_label[0] <= label[1]: # prev_label in label
                return True
            elif prev_label[0] <= label[0] <= prev_label[1]: # label in prev_label
                return True
    return False


def save_jsonl(path: str, data: List[Dict]):
    """Save data to jsonl file."""
    with jsonlines.open(path, "w") as writer:
        writer.write_all(data)


def convert_label_to_json(text, labels, start_char, end_char, targets) -> str:
    """Convert the label to json format."""
    hit_labels = [label for label in labels if start_char <= label[0] < end_char]
    hit_labels = [label for label in hit_labels if label[2] in targets["non_immu"] + targets["immu"]]
    return json.dumps({label[2]: text[label[0]:label[1]] for label in hit_labels})


def convert_to_bio(nlp: stanza.Pipeline, data: List[Dict], targets: Dict[str, List[str]]) -> pd.DataFrame:
    """Convert the data to BIO format."""
    data_bio = []
    for example in tqdm(data):
        doc = nlp(example["text"])
        tokens = [
            [
                {
                    "sentence_id": i,
                    "text": word.text,
                    "start_char": word.start_char,
                    "end_char": word.end_char,
                    "NonImmu_tag": "O",
                    "CK7_tag": "O", 
                    "TTF-1_tag": "O",
                    "CK20_tag": "O", 
                    "P40_tag": "O"
                }
                for word in sentence.words
            ]
            for i, sentence in enumerate(doc.sentences)
        ]

        # Split the tokens if `,|：|+` in the mid of the token
        special_tokens = [",", "，", "：", ":", "+", ".", "=", "?", "(", ")", "目", "-", " "]
        for i, sentence in enumerate(tokens):
            new_sentence = []
            for token in sentence:
                if any([special_token in token["text"] for special_token in special_tokens]):
                    text = []
                    start_char = token["start_char"]
                    for j, char in enumerate(token["text"]):
                        if char in special_tokens:
                            if text != []:
                                new_token = token.copy()
                                new_token["text"] = "".join(text)
                                new_token["start_char"] = start_char
                                new_token["end_char"] = start_char + len(new_token["text"])
                                new_sentence.append(new_token)
                                text = []
                                start_char = new_token["end_char"]
                            new_token = token.copy()
                            new_token["text"] = char
                            new_token["start_char"] = start_char
                            new_token["end_char"] = start_char + 1
                            new_sentence.append(new_token)
                            start_char = new_token["end_char"]
                        elif char == "x" and text != [] and text[-1].isdigit(): # 1x -> 1 x
                            new_token = token.copy()
                            new_token["text"] = "".join(text)
                            new_token["start_char"] = start_char
                            new_token["end_char"] = start_char + len(new_token["text"])
                            new_sentence.append(new_token)
                            start_char = new_token["end_char"]
                            new_token = token.copy()
                            new_token["text"] = char
                            new_token["start_char"] = start_char
                            new_token["end_char"] = start_char + 1
                            new_sentence.append(new_token)
                            start_char = new_token["end_char"]
                            text = []
                        else:
                            text.append(char)
                    if text != []:
                        new_token = token.copy()
                        new_token["text"] = "".join(text)
                        new_token["start_char"] = start_char
                        new_token["end_char"] = start_char + len(new_token["text"])
                        new_sentence.append(new_token)
                else:
                    new_sentence.append(token)
            tokens[i] = new_sentence

        for label in example["label"]:
            # Find the sentence
            sentence_index = None
            for i, sentence in enumerate(tokens):
                if sentence[0]["start_char"] <= label[0] < sentence[-1]["end_char"]:
                    sentence_index = i
                    break
            if sentence_index is None:
                print(f"Sentence not found {label} in {example['id']}")
                continue

            # Check the label begin match the token begin and modify the label begin if not match
            label_begin = label[0]
            for token in tokens[sentence_index]:
                if token["start_char"] == label[0]:
                    break
                elif token["start_char"] < label[0] < token["end_char"]:
                    label[0] = token["start_char"]
                    break

            # Check the label end match the token end and modify the label end if not match
            label_end = label[1]
            for token in tokens[sentence_index]:
                if token["end_char"] == label[1]:
                    break
                elif token["start_char"] < label[1] < token["end_char"]:
                    label[1] = token["end_char"]
                    break

            if example["text"][label[0]:label[1]] != example["text"][label_begin:label_end]:
                print(f"Label not match in {example['id']}: The original label is `{example['text'][label_begin:label_end]}`, but the final is `{example['text'][label[0]:label[1]]}`")

            # Add BIO tag
            for sentence in tokens:
                for i, token in enumerate(sentence):
                    if token["start_char"] == label[0]: # begin found
                        if label[2] in targets["immu"]:
                            token[f"{label[2]}_tag"] = f"B-{label[2]}"
                        if label[2] in targets["non_immu"]:
                            token["NonImmu_tag"] = f"B-{label[2]}"
                    elif label[0] < token["start_char"] < label[1]: # middle found
                        if label[2] in targets["immu"]:
                            token[f"{label[2]}_tag"] = f"I-{label[2]}"
                        if label[2] in targets["non_immu"]:
                            token["NonImmu_tag"] = f"I-{label[2]}"
                    elif token["end_char"] == label[1]: # end found
                        if label[2] in targets["immu"]:
                            token[f"{label[2]}_tag"] = f"I-{label[2]}"
                        if label[2] in targets["non_immu"]:
                            token["NonImmu_tag"] = f"I-{label[2]}"
                        break
        
        data_bio.extend([
            {
                "report_id": example["id"],
                "sentence_id": sentence[0]["sentence_id"],
                "text": example["text"][sentence[0]["start_char"]:sentence[-1]["end_char"]],
                "label": convert_label_to_json(
                    example["text"], example["label"], sentence[0]["start_char"], sentence[-1]["end_char"], targets
                ),
                "tokens": [token["text"] for token in sentence],
                "NonImmu_tags": [token["NonImmu_tag"] for token in sentence],
                "CK7_tags": [token["CK7_tag"] for token in sentence],
                "TTF-1_tags": [token["TTF-1_tag"] for token in sentence],
                "CK20_tags": [token["CK20_tag"] for token in sentence],
                "P40_tags": [token["P40_tag"] for token in sentence],
            }
            for sentence in tokens
        ])
    
    return data_bio


def human_prompt(text: str, targets: Dict[str, List[str]]) -> str:
    """Create ShareGPT user prompt."""
    target_list = "\n".join(targets["non_immu"] + targets["immu"])
    instruction = f"Given a sentence from a lung cancer report. Find the important information if it exist in the sentence. If the information is nonexistent, please respond `unknown`. Please respond in json format."
    return f"{instruction}\n\n### Information\n{target_list}\n\n### Sentence\n{text}\n"


def gpt_answer(label: Dict[str, str], targets: Dict[str, List[str]]) -> str:
    """Create ShareGPT gpt answer."""
    answer = {t: "unknown" for t in targets["non_immu"] + targets["immu"]}
    answer.update(json.loads(label))
    return json.dumps(answer, ensure_ascii=False)


def to_sharegpt_format(example: Dict, targets) -> Dict:
    """Convert the example to ShareGPT format."""
    return {
        "id": f"{example['report_id']}_{example['sentence_id']}",
        "conversations": [
            {"from": "human", "value": human_prompt(example["text"], targets)},
            {"from": "gpt", "value": gpt_answer(example["label"], targets)},
        ]
    }

def main():
    args = parse_args()

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data and targets
    data = load_data(args.data_path)
    data = [collect_text_label(example, args.text_name, args.label_name) for example in data]
    targets = get_targets(args.src_dir)

    # Filter reference
    data = [filter_reference(example) for example in data]

    # Clean the label
    for example in data:
        for label in example["label"]:
            label[0] = clean_head_space(example["text"], label[0])
            label[1] = clean_tail_space(example["text"], label[1])

    # Remove duplicate labels
    for example in data:
        prev_labels = []
        for label in example["label"]:
            if label not in prev_labels:
                if not check_duplicate(prev_labels, label):
                    prev_labels.append(label)
        example["label"] = prev_labels

    # Initialize stanza pipeline for tokenization
    nlp = stanza.Pipeline("en", package="mimic", processors="tokenize")

    if args.split_train_test:
        # Separate train/test
        doc_ids = list(set([example["id"] for example in data]))
        train_ids, test_ids = train_test_split(doc_ids, test_size=args.test_size, random_state=args.seed)
        train = [example for example in data if example["id"] in train_ids]
        test = [example for example in data if example["id"] in test_ids]
        print(f"Train size: {len(train)}")
        print(f"Test size: {len(test)}")

        train = convert_to_bio(nlp, train, targets)
        test = convert_to_bio(nlp, test, targets)
    
        save_jsonl(f"{args.output_dir}/train.jsonl", train)
        save_jsonl(f"{args.output_dir}/test.jsonl", test)

        # Convert to ShareGPT format
        train = [to_sharegpt_format(example, targets) for example in train]
        test = [to_sharegpt_format(example, targets) for example in test]

        Path(args.output_dir, "train_sharegpt.json").write_text(json.dumps(train, indent=2, ensure_ascii=False))
        Path(args.output_dir, "test_sharegpt.json").write_text(json.dumps(test, indent=2, ensure_ascii=False))
    else:
        data = convert_to_bio(nlp, data, targets)
        save_jsonl(f"{args.output_dir}/data.jsonl", data)

        # Convert to ShareGPT format
        data = [to_sharegpt_format(example, targets) for example in data]
        Path(args.output_dir, "data_sharegpt.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()