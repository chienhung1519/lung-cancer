from argparse import ArgumentParser, Namespace
from typing import Dict, List
import jsonlines
import pandas as pd
from tqdm.auto import tqdm
import json
from pathlib import Path
from tqdm.auto import tqdm

import stanza
from sklearn.model_selection import train_test_split


IMMU_COLUMNS = ["CK7", "TTF-1", "Napsin-A", "CK20", "P40"]
REFERENCE_PREFIX = "\nRef"
SPECIAL_TOKENS = [",", "，", "：", ":", "+", ".", "=", "?", "(", ")", "-", " "]


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--src_dir", type=str, default="./src")
    parser.add_argument("--text_name", type=str, default="text")
    parser.add_argument("--label_name", type=str, default="label")
    parser.add_argument("--split_train_test", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7687)
    args = parser.parse_args()
    return args


def load_data(data_path: str) -> List[Dict]:
    """Load data from jsonl file."""
    with jsonlines.open(data_path) as reader:
        data = [obj for obj in reader]
    return data


def filter_reference(example: Dict) -> Dict:
    """Delete the reference part in the text and label."""
    ref_index = example["text"].rfind(REFERENCE_PREFIX)
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


def convert_label_to_json(text: str, labels: List, start_char: int, end_char: int) -> str:
    """Convert the label to json format."""
    # label format: [start_char, end_char, label]
    hit_labels = [label for label in labels if start_char <= label[0] < end_char]
    return json.dumps({label[2]: text[label[0]:label[1]] for label in hit_labels})


def create_token(token, text: str, start_char: int, end_char: int) -> Dict:
    """Create token."""
    new_token = token.copy()
    new_token["text"] = text
    new_token["start_char"] = start_char
    new_token["end_char"] = end_char
    return new_token


def cut_text_by_special_tokens(sentence: List) -> List[str]:
    """Cut the text by special tokens."""
    new_sentence = []
    for token in sentence:
        if any([special_token in token["text"] for special_token in SPECIAL_TOKENS]):
            text = []
            start_char = token["start_char"]
            for char in token["text"]:
                if char in SPECIAL_TOKENS:
                    if text != []:
                        new_token = create_token(token, "".join(text), start_char, start_char + len("".join(text)))
                        new_sentence.append(new_token)
                        text = []
                        start_char = new_token["end_char"]
                    new_token = create_token(token, char, start_char, start_char + 1)
                    new_sentence.append(new_token)
                    start_char = new_token["end_char"]
                elif char == "x" and text != [] and text[-1].isdigit(): # 1x -> 1 x
                    new_token = create_token(token, "".join(text), start_char, start_char + len("".join(text)))
                    new_sentence.append(new_token)
                    start_char = new_token["end_char"]
                    new_token = create_token(token, char, start_char, start_char + 1)
                    new_sentence.append(new_token)
                    start_char = new_token["end_char"]
                    text = []
                else:
                    text.append(char)
            if text != []:
                new_sentence.append(create_token(token, "".join(text), start_char, start_char + len("".join(text))))
        else:
            new_sentence.append(token)
    return new_sentence


def find_sentence_index(doc: List, label: List, example_id: int) -> int:
    """Find the sentence where the label is in"""
    for i, sentence in enumerate(doc):
        if sentence[0]["start_char"] <= label[0] < sentence[-1]["end_char"]:
            return i
    raise(f"Sentence not found {label} in {example_id}")


def find_label_index(sentence: List, label: List, check_start: bool = False) -> int:
    for token in sentence:
        criteria = label[0] if check_start else label[1]
        if token["start_char"] <= criteria < token["end_char"]:
            return token["start_char"] if check_start else token["end_char"]
    raise(f"Label not found {label} in {sentence}")


def fit_label_to_token(doc: List, label: List, example_id: int, example_text) -> List:
    """Fit the label to the token."""
    sentence_index = find_sentence_index(doc, label, example_id)
    label_begin = find_label_index(doc[sentence_index], label, check_start=True)
    label_end = find_label_index(doc[sentence_index], label, check_start=False)
    if example_text[label[0]:label[1]] != example_text[label_begin:label_end]:
        print(f"Label not match in {example_id}: The original label is `{example_text[label_begin:label_end]}` -> the final is `{example_text[label[0]:label[1]]}`")
    return [label_begin, label_end, example_text[label[0]:label[1]]]


def convert_to_bio(data: List[Dict], targets: List) -> pd.DataFrame:
    """Convert the data to BIO format."""
    bio_data = []
    for example in tqdm(data):
        # Split the text to sentences
        nlp = stanza.Pipeline("en", package="mimic", processors="tokenize")
        doc = nlp(example["text"])

        # Initialize token list
        doc = [
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

        # Split the tokens if SPECIAL_TOKENS in the mid of the token
        doc = [cut_text_by_special_tokens(sentence) for sentence in doc]

        for label in example["label"]:
            # Check the label_begin match the token_begin and modify the label_begin if not match
            label = fit_label_to_token(doc, label, example["id"], example["text"])

            # Add BIO tag
            for sentence in doc:
                for token in sentence:
                    if token["start_char"] == label[0]: # begin found
                        if label[2] in IMMU_COLUMNS:
                            token[f"{label[2]}_tag"] = f"B-{label[2]}"
                        elif label[2] in targets:
                            token["NonImmu_tag"] = f"B-{label[2]}"
                    elif label[0] < token["start_char"] < label[1]: # middle found
                        if label[2] in IMMU_COLUMNS:
                            token[f"{label[2]}_tag"] = f"I-{label[2]}"
                        elif label[2] in targets:
                            token["NonImmu_tag"] = f"I-{label[2]}"
                    elif token["end_char"] == label[1]: # end found
                        if label[2] in IMMU_COLUMNS:
                            token[f"{label[2]}_tag"] = f"I-{label[2]}"
                        elif label[2] in targets:
                            token["NonImmu_tag"] = f"I-{label[2]}"
                        break
        
        bio_data.extend([
            {
                "report_id": example["id"],
                "sentence_id": sentence[0]["sentence_id"],
                "text": example["text"][sentence[0]["start_char"]:sentence[-1]["end_char"]],
                "label": convert_label_to_json(
                    example["text"], example["label"], sentence[0]["start_char"], sentence[-1]["end_char"]
                ),
                "tokens": [token["text"] for token in sentence],
                "NonImmu_tags": [token["NonImmu_tag"] for token in sentence],
                "CK7_tags": [token["CK7_tag"] for token in sentence],
                "TTF-1_tags": [token["TTF-1_tag"] for token in sentence],
                "CK20_tags": [token["CK20_tag"] for token in sentence],
                "P40_tags": [token["P40_tag"] for token in sentence],
            }
            for sentence in doc
        ])
    
    return bio_data


def human_prompt(text: str, targets: List) -> str:
    """Create human prompt."""
    target_text = "\n".join(targets)
    instruction = f"Given a sentence from a lung cancer report. Find the important information if it exist in the sentence. If the information is nonexistent, please respond `unknown`. Please respond in json format."
    return f"{instruction}\n\n### Information\n{target_text}\n\n### Sentence\n{text}\n"


def gpt_answer(label: Dict[str, str], targets: List) -> str:
    """Create GPT answer."""
    answer = {target: "unknown" for target in targets}
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


def convert_and_save(data: List[Dict], targets: List, output_dir: str, file_name: str):
    data = convert_to_bio(data, targets)
    save_jsonl(f"{output_dir}/{file_name}.jsonl", data)
    data_sharegpt = [to_sharegpt_format(example, targets) for example in data]
    Path(output_dir, f"{file_name}_sharegpt.json").write_text(json.dumps(data_sharegpt, indent=2, ensure_ascii=False))


def main():
    args = parse_args()

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data and resources
    data = load_data(args.data_path)
    data = [{"id": example["id"], "text": example[args.text_name], "label": example[args.label_name]} for example in data]
    targets = json.loads(Path(args.src_dir, "targets.json").read_text())

    # Clean data
    ## Filter reference
    data = [filter_reference(example) for example in data]

    ## Clean the label
    for example in data:
        for label in example["label"]:
            label[0] = clean_head_space(example["text"], label[0])
            label[1] = clean_tail_space(example["text"], label[1])

    ## Remove duplicate labels
    for example in data:
        prev_labels = []
        for label in example["label"]:
            if label not in prev_labels:
                if not check_duplicate(prev_labels, label):
                    prev_labels.append(label)
        example["label"] = prev_labels

    # Convert to BIO format
    if args.split_train_test:
        # Separate train/test
        doc_ids = list(set([example["id"] for example in data]))
        train_ids, test_ids = train_test_split(doc_ids, test_size=args.test_size, random_state=args.seed)
        train = [example for example in data if example["id"] in train_ids]
        test = [example for example in data if example["id"] in test_ids]
        print(f"Train size: {len(train)}")
        print(f"Test size: {len(test)}")

        # Convert to BIO/ShareGPT format and save
        convert_and_save(train, targets, args.output_dir, "train")
        convert_and_save(test, targets, args.output_dir, "test")
    else:
        convert_and_save(data, targets, args.output_dir, "data")
    

if __name__ == "__main__":
    main()