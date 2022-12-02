## Pattern Mining and Data Cleaning

In order to reduce the number of incomplete generations and knowledge aggregation, we implement efficient algorithms to ming linguistic patterns from data and match patterns to extract eventualities.


We adopt the stanford corenlp dependency parser to obtain text dependencies and check whether generations are valid or not,
If a sentence is common and valid, then its dependency should be highly frequent.

### Parsing

This part simply calls corenlp to parse text, which can be done by 
```bash
python generation_parser.py \
    --csv_file_name CSV_FILE_NAME \
    --raw_dir_name RAW_DIR_NAME \
    --processed_dir_name PROCESSED_DIR_NAME \
    --corenlp_path CORENLP_PATH \
    --base_corenlp_port BASE_CORENLP_PORT \
    --n_extractors N_EXTRACTORS
```
* `CSV_FILE_NAME` indicates the .csv file containing item ids and generations
* `RAW_DIR_NAME` is the folder to store raw data for the corenlp parser
* `PROCESSED_DIR_NAME` is the folder to store parsed data
* `CORENLP_PATH` is the path for corenlp, e.g., `corenlp-4.4.0`
* `BASE_CORENLP_PORT` is the initial port for corenlp server
* `N_EXTRACTORS` indicates the number of corenlp servers for parallel parsing

### Filtering

We have our prior knowledge about the pos-tags of templates (e.g., "they both are capable of" corresponds to "PRP DT VBP JJ IN"). Therefore, we can simply filter text by check the prefixes of pos-tags (which we call "meta-pattern").
Besides, we only consider highly-frequent patterns so that we add constraints for the pattern lengths (refer to Line 103-159).

We use the gSpan algorithm immplemented by [parsemis](https://github.com/timtadh/parsemis). For efficiency, we split patterns into different files and start multiple processes to obtain frequent patterns.

We provide a `pattern_filter.py` and  to combine these prodedures together:
```bash
python pattern_filter.py \
    --processed_file_name PROCESSED_FILE_NAME \
    --pattern_dir_name PATTERN_DIR_NAME \
    --relation_type RELATION_TYPE \
    --n_extractors N_EXTRACTORS
```
* `PROCESSED_FILE_NAME` indicates the .jsonl file containing parsed text
* `PATTERN_DIR_NAME` is the folder to store dependencies and frequent patterns
* `RELATION_TYPE` corresponds to the relation type for parsed text
* `N_EXTRACTORS` indicates the number of corenlp servers for parallel parsing

### Mining

After filtering, we obtain thousands of patterns. But most of them are imcomplete and redudant. Therefore, we use anothor process to get cleaner ones. In short, we check subgraph isomorphisms.
```bash
python pattern_miner.py \
    --processed_dir_name PROCESSED_DIR_NAME \
    --pattern_dir_name PATTERN_DIR_NAME \
    --relation_type RELATION_TYPE \
    --additional_tokens ADDITIONAL_TOKENS
```
* `PROCESSED_DIR_NAME` is the folder containing .jsonl files (no matter wheter they are used in the previous stage)
* `PATTERN_DIR_NAME` is the folder to store frequent patterns
* `RELATION_TYPE` corresponds to the relation type for parsed text
* `ADDITIONAL_TOKENS` indicates the number of tokens for pattern selection

### Merge

If we have already get frequent patterns for each relation, we can collect them together:
```bash
python pattern_merge.py \
  --pattern_dir_name PATTERN_DIR_NAME
  --output_file OUTPUT_FILE
```
* `PATTERN_DIR_NAME` is the folder to store frequent patterns
* `OUTPUT_FILE` corresponds to file to keep all frequent patterns

### Matching

We can apply patterns to extract eventualities (refer to [ASER](https://arxiv.org/abs/1905.00270)). Instead of checking linguistic rules, we directly apply subgraph isomorphism matching to obtain eventualities.
```bash
python pattern_match.py \
  --data_file DATA_FILE
  --pattern_file PATTERN_FILE
  --output_file OUTPUT_FILE
```
* `DATA_FILE` is the input file containing parsed data
* `PATTERN_FILE` is the pattern file
* `OUTPUT_FILE` corresponds to file to keep extracted results

## Extraction and Conceptualization

We also provide `PatternMatchEventualityExtractor` in `extractor.py` for users to extract free text.
```python
pm_event_extractor = PatternMatchEventualityExtractor(
    corenlp_path="corenlp-4.4.0",
    corenlp_port=9000,
    pattern_file="pattern/freq.txt",
)
text = "they both are related to money."
pm_event_extractor.extract_from_text(text)
```

For better knowledge aggregation, we provide `ProbaseConceptualizer` in `conceptualizer.py` for conceptualization. Please download [probase](https://www.microsoft.com/en-us/research/project/probase/) in advance.
```python
conceptulizer = ProbaseConceptualizer(probase_path="probase/data-concept-instance-relations.txt", topK=10)
text = "they both are related to money."
start_index = len(TEMPLATES["relatedTo"][1]) # 5
eventuality = pm_event_extractor.extract_from_text(text)[0]
conceptulizer.conceptualizer.conceptualize(eventuality, start_index)
```
