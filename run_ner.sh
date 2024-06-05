CUDA_VISIBLE_DEVICES=1 python run_ner.py \
  --train_file data/processed/lung-cancer-0001-3042/train.jsonl \
  --test_file data/processed/lung-cancer-0001-3042/test.jsonl \
  --ntu_file data/processed/ntuh2tmuh-manual-rewritten/data.jsonl \
  --output_dir outputs/gatortron-base \
  --model_name_or_path UFNLP/gatortron-base

CUDA_VISIBLE_DEVICES=0 python run_ner.py \
  --train_file data/processed/lung-cancer-0001-3042/train.jsonl \
  --test_file data/processed/lung-cancer-0001-3042/test.jsonl \
  --ntu_file data/processed/ntuh2tmuh-manual-rewritten/data.jsonl \
  --output_dir outputs/bert-base \
  --model_name_or_path bert-base-uncased

CUDA_VISIBLE_DEVICES=0 python run_ner.py \
  --train_file data/processed/lung-cancer-0001-3042/train.jsonl \
  --test_file data/processed/lung-cancer-0001-3042/test.jsonl \
  --ntu_file data/processed/ntuh2tmuh-manual-rewritten/data.jsonl \
  --output_dir outputs/clinical-bert \
  --model_name_or_path emilyalsentzer/Bio_ClinicalBERT