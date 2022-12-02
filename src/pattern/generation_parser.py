import argparse
import json
import os
import shutil
import pickle
import multiprocessing
import math
import pandas as pd
from collections import Counter, defaultdict
from itertools import chain
from tqdm import tqdm
from utils import *


def process_raw_file(
    raw_path, processed_path=None, corenlp_path="corenlp-4.4.0", corenlp_port=9000, annotators=None, max_len=None
):
    """ Process a file that contains raw texts

    :param raw_path: the file name to a file that contains raw texts
    :type raw_path: str
    :param processed_path: the file name to a file to store parsed results
    :type processed_path: str or None
    :param corenlp_path: the path to the Stanford CoreNLP package
    :type corenlp_path: str
    :param corenlp_port: the port number of the Stanford CoreNLP server
    :type corenlp_port: int
    :param annotators: the annotators to use
    :type annotators: list or None
    :param max_len: the maximum length of a paragraph
    :type max_len: int or None
    :return: the parsed results of the given file
    :rtype: List[List[Dict[str, object]]]
    """

    if max_len is None:
        max_len = MAX_LEN
    corenlp_client, _ = get_corenlp_client(corenlp_path=corenlp_path, corenlp_port=corenlp_port, annotators=annotators)

    paragraphs = []
    sids = []
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = json.loads(line)
            paragraphs.append(line["text"])
            sids.append(line["sid"])

    sid = 1
    para_lens = []
    for i in range(len(paragraphs)):
        paragraphs[i] = parse_sentense_with_stanford(paragraphs[i], corenlp_client, annotators, max_len)
        para_lens.append(len(paragraphs[i]) + sid)
        for sent in paragraphs[i]:
            sent["sid"] = f"{sids[i]}|{sid}"
            sid += 1

    if processed_path is not None:
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"sentence_lens": para_lens}))
            f.write("\n")
            for para in paragraphs:
                for sent in para:
                    f.write(json.dumps(sent))
                    f.write("\n")
    return paragraphs


def load_processed_data(processed_path):
    """ This method retrieves all paragraphs from a processed file

    :type processed_path: str or None
    :param processed_path: the file path of the processed file
    :return: a list of lists of dicts
    """
    with open(processed_path, "r") as f:
        sent_len = json.loads(f.readline())["sentence_lens"]
        paragraphs = list()
        line_no = 1
        para_idx = 0
        while para_idx < len(sent_len):
            paragraph = list()
            end_no = sent_len[para_idx]
            while line_no < end_no:
                sent = json.loads(f.readline())
                if "sid" not in sent:
                    sent["sid"] = processed_path + "|" + str(line_no)
                paragraph.append(sent)
                line_no += 1
            para_idx += 1
            paragraphs.append(paragraph)
    return paragraphs


def process_file(
    raw_path=None,
    processed_path=None,
    corenlp_path="corenlp-4.4.0",
    corenlp_port=9000,
):
    if os.path.exists(processed_path):
        processed_data = load_processed_data(processed_path)
    elif os.path.exists(raw_path):
        processed_data = process_raw_file(
            raw_path, processed_path, corenlp_path=corenlp_path, corenlp_port=corenlp_port
        )
    else:
        raise ValueError("Error: at least one of raw_path and processed_path should not be None.")
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_name", type=str, default="elec.csv")
    parser.add_argument("--raw_dir_name", type=str, default="raw")
    parser.add_argument("--processed_dir_name", type=str, default="parse")
    parser.add_argument("--corenlp_path", type=str, default="corenlp-4.4.0")
    parser.add_argument("--base_corenlp_port", type=int, default=9000)
    parser.add_argument("--n_extractors", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.raw_dir_name, exist_ok=True)
    os.makedirs(args.processed_dir_name, exist_ok=True)

    df = pd.read_csv(
        args.csv_file_name, usecols=["item_a_id", "relation", "item_b_id", "assertion"]
    )
    data = defaultdict(list)
    for row in df.itertuples():
        if row.relation not in TEMPLATES:
            continue
        prefix = TEMPLATES[row.relation][0]
        if len(prefix) == 0:
            idx = row.assertion.index("because ") + 8
        else:
            assert prefix in row.assertion
            idx = row.assertion.index(prefix)
        sid = "{}-{}-{}|{}".format(row.item_a_id, row.relation, row.item_b_id, row.Index)
        data[row.relation].append({"sid": sid, "text": row.assertion[idx:]})

    raw_paths, processed_paths = list(), list()
    file_name_prefix = os.path.splitext(os.path.basename(args.csv_file_name))[0]
    for relation in sorted(data.keys()):
        raw_path = os.path.join(args.raw_dir_name, relation + "_" + file_name_prefix + ".jsonl")
        processed_path = os.path.join(args.processed_dir_name, relation + "_" + file_name_prefix + ".jsonl")
        with open(raw_path, "w") as f:
            for x in data[relation]:
                f.write(json.dumps(x))
                f.write("\n")
        raw_paths.append(raw_path)
        processed_paths.append(processed_path)

    with multiprocessing.Pool(args.n_extractors) as pool:
        results = list()
        for i in range(len(raw_paths)):
            extractor_idx = i % args.n_extractors
            corenlp_port = args.base_corenlp_port + extractor_idx
            results.append(
                pool.apply_async(
                    process_file, args=(raw_paths[i], processed_paths[i], args.corenlp_path, corenlp_port)
                )
            )
        pool.close()
        for x in tqdm(results):
            x.get()
