proj_dir="/home"
data_path=$proj_dir"/data"
probase_path=$proj_dir"/probase/data-concept-instance-relations.txt"
declare -a relations=("open" "relatedTo" "isA" "partOf" "madeOf" "similarTo" "createdBy" "hasA" "propertyOf" "distinctFrom" "usedFor" "can" "capableOf" "definedAs" "symbolOf" "mannerOf" "deriveFrom" "effect" "cause" "motivatedBy" "causeEffect")

data_file_suffix="quality_annotation_all.jsonl"
processed_dir_name=$data_path"/parse"
pattern_dir_name=$data_path"/pattern"
extraction_dir_name=$data_path"/extraction"
conceptualization_dir_name=$data_path"/conceptualization"

mkdir -p $extraction_dir_name

for relation in ${relations[@]}
do
    python src/pattern/pattern_match.py \
        --data_file $extraction_dir_name"/"$relation"_"$data_file_suffix \
        --pattern_file $pattern_dir_name"/"$relation"-freq.txt" \
        --relation_type $relation \
        --probase_path $probase_path \
        --output_file $conceptualization_dir_name"/"$relation"_"$data_file_suffix
done
