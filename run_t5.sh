python run_t5.py \
  --train_file ./data/processed/lung-cancer-0001-3042/train.jsonl \
  --test_file ./data/processed/lung-cancer-0001-3042/test.jsonl \
  --ntu_file ./data/processed/ntuh2tmuh/data.jsonl \
  --output_dir ./outputs/t5-base \
  --model_name_or_path google/t5-v1_1-base \
  --n_gpu 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 8

python run_t5.py \
  --train_file ./data/processed/lung-cancer-0001-3042/train.jsonl \
  --test_file ./data/processed/lung-cancer-0001-3042/test.jsonl \
  --ntu_file ./data/processed/ntuh2tmuh/data.jsonl \
  --output_dir ./outputs/clinical-t5-base \
  --model_name_or_path luqh/ClinicalT5-base \
  --n_gpu 1 \
  --from_flax \
  --batch_size 1 \
  --gradient_accumulation_steps 8