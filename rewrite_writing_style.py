from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import jsonlines
from typing import Dict, List
import random
from transformers import pipeline
from tqdm.auto import tqdm
import torch
from unsloth import FastLanguageModel


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--source_data_path", type=str, required=True)
    parser.add_argument("--target_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--src_dir", type=str, default="./src")
    parser.add_argument("--num_per_target", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--seed", type=int, default=7687)
    args = parser.parse_args()
    return args


def load_data(data_path: str) -> List[Dict]:
    """Load data from jsonl file."""
    with jsonlines.open(data_path) as reader:
        data = [obj for obj in reader]
    return data


def random_select_data(data: List[Dict], num_per_target: int, targets: Dict[str, List]) -> List[str]:
    """Random select data from source data according to targets."""
    sampled_data = []
    for target in targets["immu"] + targets["non_immu"]:
        data_with_target = []
        for example in data:
            if f"B-{target}" in example["NonImmu_tags"] + example["CK7_tags"] + example["TTF-1_tags"] + example["CK20_tags"] + example["P40_tags"]:
                data_with_target.append(example["text"].replace("\n", " "))
        sampled_data.extend(random.sample(data_with_target, num_per_target))
    return sampled_data


def to_prompt(source_data_text: str, target_data_text: str, sentence: str) -> str:
    """Convert data to prompt."""
    return f"""\
Given a sentence from clinical note in A hospital, \
please rewrite the text into the writing style of B hospital. \
It is important to keep the semantic meaning unchanged. \
Respond in JSON with key of `text`.

### B hospital sentence examples:
{target_data_text}

### Sentence to rewrite:
{sentence}

### Rewritten Sentence

"""


def extract_json_from_text(response: str) -> str:
    """Extract JSON from response. Using regex to extract JSON {}."""
    try:
        json_response = response.split("{")[1].split("}")[0]
        json_response = "{" + json_response + "}"
        json_response = json.loads(json_response)
        return json_response["text"]
    except:
        print(response)
        return response
    

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
            {"from": "human", "value": human_prompt(example["rewritten_text"], targets)},
            {"from": "gpt", "value": gpt_answer(example["label"], targets)},
        ]
    }


def main():
    # Parse arguments
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    random.seed(args.seed)

    # Read data
    source_data = load_data(args.source_data_path)
    target_data = load_data(args.target_data_path)
    targets = json.loads(Path(args.src_dir, "targets.json").read_text())

    # Random select source data
    sampled_source_data = random_select_data(source_data, args.num_per_target, targets)
    sampled_source_data_text = "\n".join(sampled_source_data)
    
    # Random select target data
    sampled_target_data = random_select_data(target_data, args.num_per_target, targets)
    sampled_target_data_text = "\n".join(sampled_target_data)

    Path(args.output_dir, "sampled_data.json").write_text(
        "Sampled Source Data:\n" + sampled_source_data_text + 
        "\n\nSampled Target Data:\n" + sampled_target_data_text
    )

    # Rewrite data using transformers pipeline
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 4096,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    for example in tqdm(target_data):
        prompt = to_prompt(sampled_source_data_text, sampled_target_data_text, example["text"].replace("\n", " "))
        rewite_text = generator(
            prompt, 
            return_full_text=False, 
            max_new_tokens=1024, 
            repetition_penalty=1.0, 
            temperature=0.7,
            eos_token_id=tokenizer.encode('\n')
        )[0]["generated_text"]
        example["rewritten_text"] = rewite_text.split("\n")[0].strip()

        with jsonlines.open(output_dir / "data.jsonl", "a") as writer:
            writer.write(example)

    # Generate ShareGPT data
    results = load_data(output_dir / "data.jsonl")
    sharegpt_data = [to_sharegpt_format(result, targets) for result in results]
    Path(args.output_dir, "data_sharegpt.json").write_text(json.dumps(sharegpt_data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()