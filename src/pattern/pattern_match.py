import argparse
import json
import igraph as ig
import numpy as np
import os
from pprint import pprint
from collections import Counter
from pattern_retriever import PatternRetriever
from pattern_miner import *
from utils import *
from object import Eventuality


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="parse/capableOf_elec.jsonl")
    parser.add_argument("--pattern_file", type=str, default="pattern/capableOf-freq.txt")
    parser.add_argument("--output_file", type=str, default="extraction/capableOf_elec.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # load patterns
    pattern_retriever = PatternRetriever()
    # pprint(meta_patterns)
    patterns = list()
    for pattern in read_patterns(args.pattern_file, fuzzy=True):
        pattern_g = construct_igraph(pattern[0], pattern[1])
        patterns.append((pattern[0], pattern[1], pattern_g))
    print(f"{len(patterns)} patterns loaded")

    # remove redundancies
    patterns.sort(key=lambda x: (len(x[1]), len(x[0]), x[0], x[1]), reverse=True)
    i = 0
    j = len(patterns) - 1
    duplicate_indices = set()
    for i in range(len(patterns)):
        if i in duplicate_indices:
            continue
        for j in range(i + 1, len(patterns)):
            if j in duplicate_indices:
                continue
            sub_iso = pattern_retriever.get_subisomorphisms(patterns[j][2], patterns[i][2])
            if len(sub_iso) > 0:
                duplicate_indices.add(j)
    patterns = [pattern for idx, pattern in enumerate(patterns) if idx not in duplicate_indices]
    patterns.sort(key=lambda x: (len(x[1]), len(x[0]), x[0], x[1]), reverse=True)
    print(f"{len(patterns)} patterns after removing redundancies")
    pattern_hierarchy = build_hierarchy(patterns)

    # load data
    with open(args.data_file, "r", encoding="utf-8") as f:
        lines = list()
        for line in f:
            line = json.loads(line)
            if "dependencies" in line and "pos_tags" in line and "tokens" in line:
                line["graph"] = construct_igraph(line["pos_tags"], line["dependencies"])
                line["eventualities"] = []
                lines.append(line)
    print(f"{len(lines)} lines loaded")
    print()

    len_cum = compute_cumulative_function(Counter([len(line["tokens"]) for line in lines]))
    N = len(lines)
    pattern_match_flag = [(1 << len(patterns))] * len(lines)
    freq_patterns = Counter()  # only pos tags in the correct order are recorded
    pattern_ctr = [0] * len(patterns)
    line_used_indices = set()

    for pattern_idx, pattern in enumerate(patterns):
        flag = (1 << pattern_idx)
        pattern_g = pattern[2]

        template = None
        example = None
        current_indices = list()

        for line_idx, line in enumerate(lines):
            # if line_idx in line_used_indices:
            #     continue
            rel = line["sid"].split("-")[-2]
            rel_len1 = len(TEMPLATES[rel][1].split())

            if pattern_match_flag[line_idx] & flag != 0:
                current_indices.append(line_idx)
                pattern_ctr[pattern_idx] += 1

            elif pattern_match_flag[line_idx] ^ (1 << len(patterns)) == 0 and len(line["pos_tags"]) >= len(
                pattern[0]
            ) and len(line["dependencies"]) >= len(pattern[1]):
                parsed_g = line["graph"]  # construct_igraph(line["pos_tags"], line["dependencies"])
                # if parsed_g.vcount() - pattern_g.vcount() > 10:
                # parsed_g = parsed_g.induced_subgraph(list(range(pattern_g.vcount() + 10)), "create_from_scratch")
                subisos = pattern_retriever.get_subisomorphisms(parsed_g, pattern_g)

                sep_indices = []
                for pos_tag_idx, pos_tag in enumerate(line["pos_tags"]):
                    if pos_tag == "WRB" or pos_tag.startswith("WP"):
                        sep_indices.append(pos_tag_idx)
                    elif pos_tag in [";", ".", "?", "!"]:
                        sep_indices.append(pos_tag_idx)
                for subiso in subisos:
                    match_failed = False

                    for sep_idx in sep_indices:
                        for idx1 in subiso:
                            for idx2 in subiso:
                                if idx1 == idx2:
                                    continue
                                if (idx1 - sep_idx) * (idx2 - sep_idx) <= 0:
                                    match_failed = True
                                    break
                            if match_failed:
                                break
                        if match_failed:
                            break
                    if match_failed:
                        continue

                    current_indices.append(line_idx)
                    pattern_ctr[pattern_idx] += 1

                    subiso_set = set(subiso)
                    optional_indices = set()
                    optional_cnt = -1

                    while len(optional_indices) != optional_cnt:
                        optional_cnt = len(optional_indices)
                        for dep in line["dependencies"]:
                            if (dep[0] in subiso_set or dep[0] in optional_indices) and \
                            (dep[2] not in subiso_set or dep[2] not in optional_indices) and \
                            line["pos_tags"][dep[2]] in OPTIONAL_POS_TAGS:
                                optional_indices.add(dep[2])

                    selected_edges = list()
                    skeleton_dependencies = list()
                    for dep in line["dependencies"]:
                        if dep[0] in subiso_set and dep[2] in subiso_set:
                            selected_edges.append(dep)
                            skeleton_dependencies.append(dep)
                        elif dep[0] in subiso_set and dep[2] in optional_indices:
                            selected_edges.append(dep)
                        elif dep[0] in optional_indices and dep[2] in optional_indices:
                            selected_edges.append(dep)

                    eventuality = Eventuality(
                        pattern=construct_pattern(pattern_g, parsed_g, subiso),
                        dependencies=selected_edges,
                        skeleton_dependencies=skeleton_dependencies,
                        parsed_result=line
                    ).to_dict()
                    duplicated = False
                    eidx = len(line["eventualities"]) - 1
                    while eidx >= 0:
                        if line["eventualities"][eidx]["eid"] == eventuality["eid"]:
                            duplicated = True
                            break
                        eidx -= 1
                    if not duplicated:
                        line["eventualities"].append(eventuality)

                    if template is None:
                        template = [pattern[0][i] for i in np.argsort(subiso)]
                        example = line
                    elif len(line["pos_tags"]) < len(example["pos_tags"]):  # shorter
                        template = [pattern[0][i] for i in np.argsort(subiso)]
                        example = line
                    elif len(line["pos_tags"]) == len(
                        example["pos_tags"]
                    ) and sum(subiso) / len(subiso) < (len(example["pos_tags"]) - 1) / 2:  # closer
                        template = [pattern[0][i] for i in np.argsort(subiso)]
                        example = line

        if template is None and len(current_indices) > 0:
            for line_idx in sorted(current_indices, key=lambda line_idx: len(lines[line_idx])):
                line = lines[line_idx]
                subisos = pattern_retriever.get_subisomorphisms(parsed_g, pattern_g)
                for subiso in subisos:
                    if check_match_with_prefix(subiso, rel_len1) and check_match_with_must(subiso, parsed_g):
                        template = [pattern[0][i] for i in np.argsort(subiso)]
                        example = line
                        break
                if template is not None:
                    break

        if template is None:  # invalid pattern
            pattern_ctr[pattern_idx] = 0
            continue

        # if pattern_ctr[pattern_idx] < N * 1e-4: # low frequent
        #     pattern_ctr[pattern_idx] = 0
        #     continue

        freq_pattern = None
        parsed_g = construct_igraph(example["pos_tags"], example["dependencies"])
        subisos = pattern_retriever.get_subisomorphisms(parsed_g, pattern_g)
        for subiso in subisos:
            if check_match_with_prefix(subiso, rel_len1) and check_match_with_must(subiso, parsed_g):
                freq_pattern = construct_pattern(pattern_g, parsed_g, subiso)
                if freq_pattern[0][-1] not in FORBIDDED_END_POS_TAGS and check_connect(freq_pattern):
                    break
                else:
                    freq_pattern = None

        if freq_pattern is None:
            pattern_ctr[pattern_idx] = 0
            continue

        freq_patterns[freq_pattern] += pattern_ctr[pattern_idx]
        for child_idx in pattern_hierarchy[pattern_idx].children:
            try:
                flag |= (1 << child_idx)
            except:
                tmp = (1 << child_idx)
                tmp = int(flag) | int(tmp)
                flag = tmp
        for line_idx in current_indices:
            pattern_match_flag[line_idx] |= flag
        line_used_indices.update(current_indices)

        print_sent(example, subiso)
        print(
            f"pattern {template} with {len(template)} tokens for relation {rel} has been matched {pattern_ctr[pattern_idx]} times"
        )
        print("-" * 80)
    print("=" * 80)
    print(
        "obtain {} frequent patterns ({:2.3f} are covered)".format(
            len(freq_patterns), 100 * len(line_used_indices) / N
        )
    )
    with open(args.output_file, "w") as f:
        for line in lines:
            line.pop("graph")
            f.write(json.dumps(line))
            f.write("\n")
