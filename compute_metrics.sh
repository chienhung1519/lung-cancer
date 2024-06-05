python compute_metrics.py \
  --predictions_file outputs/bert-base/predict/ntu/generated_predictions.jsonl \
  --outputs_dir outputs/bert-base/predict/ntu \
  --model_type bert

python compute_metrics.py \
  --predictions_file outputs/bert-base/predict/tmu/generated_predictions.jsonl \
  --outputs_dir outputs/bert-base/predict/tmu \
  --model_type bert

python compute_metrics.py \
  --predictions_file outputs/clinical-bert/predict/ntu/generated_predictions.jsonl \
  --outputs_dir outputs/clinical-bert/predict/ntu \
  --model_type bert

python compute_metrics.py \
  --predictions_file outputs/clinical-bert/predict/tmu/generated_predictions.jsonl \
  --outputs_dir outputs/clinical-bert/predict/tmu \
  --model_type bert

python compute_metrics.py \
  --predictions_file outputs/gatortron-base/predict/ntu/generated_predictions.jsonl \
  --outputs_dir outputs/gatortron-base/predict/ntu \
  --model_type bert

python compute_metrics.py \
  --predictions_file outputs/gatortron-base/predict/tmu/generated_predictions.jsonl \
  --outputs_dir outputs/gatortron-base/predict/tmu \
  --model_type bert

python compute_metrics.py \
  --predictions_file outputs/t5-base/predict/ntu/generated_predictions.jsonl \
  --outputs_dir outputs/t5-base/predict/ntu \
  --model_type t5

python compute_metrics.py \
  --predictions_file outputs/t5-base/predict/tmu/generated_predictions.jsonl \
  --outputs_dir outputs/t5-base/predict/tmu \
  --model_type t5

python compute_metrics.py \
  --predictions_file outputs/clinical-t5-base/predict/ntu/generated_predictions.jsonl \
  --outputs_dir outputs/clinical-t5-base/predict/ntu \
  --model_type t5

python compute_metrics.py \
  --predictions_file outputs/clinical-t5-base/predict/tmu/generated_predictions.jsonl \
  --outputs_dir outputs/clinical-t5-base/predict/tmu \
  --model_type t5

python compute_metrics.py \
  --predictions_file ./outputs/LLaMA3-8B-Instruct/loraplus/predict/tmu/generated_predictions.jsonl \
  --outputs_dir ./outputs/LLaMA3-8B-Instruct/loraplus/predict/tmu/ \
  --model_type llama

python compute_metrics.py \
  --predictions_file ./outputs/LLaMA3-8B-Instruct/loraplus/predict/ntu_manual_rewritten/generated_predictions.jsonl \
  --outputs_dir ./outputs/LLaMA3-8B-Instruct/loraplus/predict/ntu_manual_rewritten/ \
  --model_type llama