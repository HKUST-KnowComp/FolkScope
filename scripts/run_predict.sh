export TOKENIZERS_PARALLELISM=false

pretrained_lm="pretrained-lm"
data_path="/home/data"

model_parameter="classifier_quality/deberta-v3-large-1e-5-42-16"
seed="42"

CUDA_VISIBLE_DEVICES=1 python src/classifier/run_classification.py \
    --model_name_or_path  $data_path/$model_parameter \
    --test_file  $data_path/quality_annotation_all.csv \
    --do_predict \
    --max_seq_length 256 \
    --per_device_eval_batch_size 32 \
    --logging_steps 100 \
    --max_predict_samples 20000000 \
    --output_dir $data_path/$model_parameter \
    --overwrite_output 
 