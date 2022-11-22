export TOKENIZERS_PARALLELISM=false

proj_dir=/home/

CUDA_VISIBLE_DEVICES=0,1 python src/generator/generation.py \
  --data_dir $proj_dir/data \
  --output_dir $proj_dir/output \
  --model_name_or_path /home/data/pretrained-lm/opt-30b \
  --model_type opt30b \
  --max_seq_length 80 \
  --behavior_type 'cobuy' \
  --generation_type "inference" \
  --prompt_type "prefix" \
  --max_generation_length 100 \
  --num_generated_sent 3 \
  --batch_size 10 \
  --category "Clothing,-Shoes-&-Jewelry" \
  --num_samples 100000 \
  --run_index 0