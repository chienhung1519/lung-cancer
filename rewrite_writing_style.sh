CUDA_VISIBLE_DEVICES=1 python rewrite_writing_style.py \
    --source_data_path ./data/processed/lung-cancer-0001-3042/train.jsonl \
    --target_data_path ./data/processed/ntuh2tmuh/data.jsonl \
    --output_dir ./data/processed/ntuh2tmuh-rewritten