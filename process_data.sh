# python process_data.py \
#   --data_path ./data/raw/lung-cancer-0001-3042.jsonl \
#   --output_dir ./data/processed/lung-cancer-0001-3042 \
#   --split_train_test

# python process_data.py \
#   --data_path ./data/raw/ntuh2tmuh.jsonl \
#   --output_dir ./data/processed/ntuh2tmuh

python process_data.py \
  --data_path ./data/raw/ntuh2tmuh.jsonl \
  --output_dir ./data/processed/ntuh2tmuh-manual_rewritten \
  --text_name text_II \
  --label_name label_II