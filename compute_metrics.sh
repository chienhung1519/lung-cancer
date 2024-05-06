python compute_metrics.py \
  --predictions_file ./outputs/LLaMA3-8B-Instruct/loraplus/predict/tmu/generated_predictions.jsonl \
  --outputs_dir ./outputs/LLaMA3-8B-Instruct/loraplus/predict/tmu/

python compute_metrics.py \
  --predictions_file ./outputs/LLaMA3-8B-Instruct/loraplus/predict/ntu/generated_predictions.jsonl \
  --outputs_dir ./outputs/LLaMA3-8B-Instruct/loraplus/predict/ntu/

python compute_metrics.py \
  --predictions_file ./outputs/LLaMA3-8B-Instruct/loraplus/predict/ntu_manual_rewritten/generated_predictions.jsonl \
  --outputs_dir ./outputs/LLaMA3-8B-Instruct/loraplus/predict/ntu_manual_rewritten/

python compute_metrics.py \
  --predictions_file ./outputs/LLaMA3-8B/loraplus/predict/tmu/generated_predictions.jsonl \
  --outputs_dir ./outputs/LLaMA3-8B/loraplus/predict/tmu/

python compute_metrics.py \
  --predictions_file ./outputs/LLaMA3-8B/loraplus/predict/ntu/generated_predictions.jsonl \
  --outputs_dir ./outputs/LLaMA3-8B/loraplus/predict/ntu/