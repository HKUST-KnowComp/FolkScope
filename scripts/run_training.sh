export TOKENIZERS_PARALLELISM=false

pretrained_lm="pretrained-lm"
data_path="/home/data"

lrs=("1e-5" "5e-6" "3e-6" "1e-6")
lrs=("5e-6" "1e-6")
bsz=("16") 

models=("deberta-v3-large" "deberta-v3-base")
seed="42"

for model in ${models[@]}
do 
    for lr in ${lrs[@]}
    do
      for bs in ${bsz[@]}
      do 
        CUDA_VISIBLE_DEVICES=0 python src/classifier/run_classification.py \
          --model_name_or_path  $pretrained_lm/$model  \
          --train_file $data_path/quality_large_annotation_train.csv \
          --validation_file $data_path/quality_large_annotation_dev.csv \
          --do_train \
          --do_eval \
          --max_seq_length 256 \
          --per_device_train_batch_size $bs \
          --per_device_eval_batch_size 16 \
          --learning_rate $lr \
          --gradient_accumulation_steps 2 \
          --num_train_epochs 10 \
          --logging_steps 100 \
          --seed $seed \
          --save_steps 500 \
          --save_total_limit 2 \
          --load_best_model_at_end \
          --metric_for_best_model "f1" \
          --evaluation_strategy "steps" \
          --eval_steps 500 \
          --report_to wandb \
          --run_name $model-$lr-$seed-$bs-quality-large \
          --output_dir $data_path/classifier_quality/$model-$lr-$seed-$bs-large \
          --overwrite_output
        done
      done
done
