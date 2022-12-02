proj_dir="/home"
data_path=$proj_dir"/data"
corenlp_path=$proj_dir"/stanford-corenlp-4.4.0"
num_workers=10
declare -a relations=("open" "relatedTo" "isA" "partOf" "madeOf" "similarTo" "createdBy" "hasA" "propertyOf" "distinctFrom" "usedFor" "can" "capableOf" "definedAs" "symbolOf" "mannerOf" "deriveFrom" "effect" "cause" "motivatedBy" "causeEffect")

csv_file_name=$data_path"/quality_annotation_all.csv"
raw_dir_name=$data_path"/raw"
processed_dir_name=$data_path"/parse"
pattern_dir_name=$data_path"/pattern"

mkdir -p $raw_dir_name
mkdir -p $processed_dir_name
mkdir -p $pattern_dir_name

python src/pattern/generation_parser.py \
    --csv_file_name $csv_file_name \
    --raw_dir_name $raw_dir_name \
    --processed_dir_name $processed_dir_name \
    --corenlp_path $corenlp_path \
    --n_extractors $num_workers

for relation in ${relations[@]}
do
    python src/pattern/pattern_filter.py \
        --processed_file_name $processed_dir_name \
        --pattern_dir_name $pattern_dir_name \
        --relation_type $relation \
        --n_extractors $num_workers
done

for relation in ${relations[@]}
do
    python src/pattern/pattern_miner.py \
        --processed_file_name $processed_dir_name \
        --pattern_dir_name $pattern_dir_name \
        --relation_type $relation
done

python src/pattern/pattern_merge.py \
    --pattern_dir_name $pattern_dir_name \
    --output_file $pattern_dir_name"/freq.txt"
