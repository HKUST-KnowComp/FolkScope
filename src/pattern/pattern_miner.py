import argparse
import json
import igraph as ig
import numpy as np
import os
from pprint import pprint
from collections import Counter
from termcolor import colored
from tqdm import tqdm
import bisect

from pattern_retriever import PatternRetriever

from utils import *

pattern_retriever = PatternRetriever()


def read_patterns(filename, fuzzy=False, return_frequency=False):
    patterns = list()
    pattern_freqs = list()
    with open(filename, "r", encoding="utf-8") as f:
        vid2label = dict()
        edges = list()
        freq = 0
        for line in f:
            if line.startswith("t #"):
                freq = int(line.split("#")[1].strip())
                if len(vid2label) > 0:
                    vid_to_remove = []
                    for vid, label in vid2label.items():
                        if label in PUNCTUATION_SET:
                            vid_to_remove.append(vid)
                    if len(vid_to_remove) > 0:
                        vid_to_remove.sort(reverse=False)
                        new_vid2label = dict()
                        for v in range(len(vid2label)):
                            offset = bisect.bisect_left(vid_to_remove, v)
                            if offset < len(vid_to_remove) and vid_to_remove[offset] == v:
                                continue
                            new_vid2label[v - offset] = vid2label[v]
                        new_edges = []
                        for u, elabel, v in edges:
                            u_offset = bisect.bisect_left(vid_to_remove, u)
                            if u_offset < len(vid_to_remove) and vid_to_remove[u_offset] == u:
                                continue
                            v_offset = bisect.bisect_left(vid_to_remove, v)
                            if v_offset < len(vid_to_remove) and vid_to_remove[v_offset] == v:
                                continue
                            new_edges.append((u - u_offset, elabel, v - v_offset))
                        vid2label = new_vid2label
                        edges = new_edges
                    pattern = (tuple([vid2label[v] for v in range(len(vid2label))]), tuple(edges))
                    patterns.append(pattern)
                    pattern_freqs.append(freq)
                vid2label = dict()
                edges = list()
            elif line.startswith("v"):
                vid, vlabel = line[2:-1].split(" ")
                if fuzzy:
                    if vlabel.startswith("NN"):
                        vlabel = "NN"
                    elif vlabel.startswith("RB"):
                        vlabel = "RB"
                    elif vlabel.startswith("JJ"):
                        vlabel = "JJ"
                vid2label[int(vid)] = vlabel
            elif line.startswith("e"):
                u, v, elabel = line[2:-1].split(" ")
                elabel = elabel.split(":")[0]
                edges.append((int(u), elabel, int(v)))  # list of (u, elabel, v)
        if len(vid2label) > 0:
            vid_to_remove = []
            for vid, label in vid2label.items():
                if label in PUNCTUATION_SET:
                    vid_to_remove.append(vid)
            if len(vid_to_remove) > 0:
                vid_to_remove.sort(reverse=False)
                new_vid2label = dict()
                for v in range(len(vid2label)):
                    offset = bisect.bisect_left(vid_to_remove, v)
                    if offset < len(vid_to_remove) and vid_to_remove[offset] == v:
                        continue
                    new_vid2label[v - offset] = vid2label[v]

                new_edges = []
                for u, elabel, v in edges:
                    u_offset = bisect.bisect_left(vid_to_remove, u)
                    if u_offset < len(vid_to_remove) and vid_to_remove[u_offset] == u:
                        continue
                    v_offset = bisect.bisect_left(vid_to_remove, v)
                    if v_offset < len(vid_to_remove) and vid_to_remove[v_offset] == v:
                        continue
                    new_edges.append((u - u_offset, elabel, v - v_offset))
                vid2label = new_vid2label
                edges = new_edges
            pattern = (tuple([vid2label[v] for v in range(len(vid2label))]), tuple(edges))
            patterns.append(pattern)
            pattern_freqs.append(freq)

    if return_frequency:
        return patterns, pattern_freqs
    else:
        return patterns


def skeletonize_pattern(pattern, template_indices):
    template_indices = set(template_indices)
    if isinstance(pattern, (tuple, list)):
        if not (len(pattern) == 2 or len(pattern) == 3):
            raise ValueError

        if isinstance(pattern[0], dict) and isinstance(pattern[1], (tuple, list)):  # vid2label, edges
            skeleton_indices = []
            for idx, x in pattern[0].items():
                if idx in template_indices or (x not in OPTIONAL_POS_TAGS and x not in IGNORE_POS_TAGS):
                    skeleton_indices.append(idx)
            if len(skeleton_indices) < len(pattern[0]):
                indices_reverse = {k: -1 for k in pattern[0].keys()}
                skeleton_indices.sort()
                for i, j in enumerate(skeleton_indices):
                    indices_reverse[j] = i
                skeleton_edges = []
                for dep in pattern[1]:
                    u, elabel, v = dep
                    u, v = indices_reverse[u], indices_reverse[v]
                    if u != -1 and v != -1:
                        skeleton_edges.append((u, elabel, v))
                if len(pattern) == 2:
                    pattern = (
                        dict(zip(range(len(skeleton_indices)),
                                 [pattern[0][k] for k in skeleton_indices])), tuple(skeleton_edges)
                    )
                elif len(pattern) == 3:
                    pattern = (
                        dict(zip(range(len(skeleton_indices)),
                                 [pattern[0][k] for k in skeleton_indices])), tuple(skeleton_edges)
                    )
                    pattern = pattern + (construct_igraph(pattern[0], pattern[1]), )
        elif isinstance(pattern[0], (tuple, list)) and isinstance(pattern[1], (tuple, list)):  # pos_tags, edges
            skeleton_indices = []
            for idx, x in enumerate(pattern[0]):
                if idx in template_indices or (x not in OPTIONAL_POS_TAGS and x not in IGNORE_POS_TAGS):
                    skeleton_indices.append(idx)
            if len(skeleton_indices) < len(pattern[0]):
                n = len(pattern[0])
                indices_reverse = [-1] * n
                for i, j in enumerate(skeleton_indices):
                    indices_reverse[j] = i
                skeleton_edges = []
                for dep in pattern[1]:
                    u, elabel, v = dep
                    u, v = indices_reverse[u], indices_reverse[v]
                    if u != -1 and v != -1:
                        skeleton_edges.append((u, elabel, v))
                if len(pattern) == 2:
                    pattern = (tuple([pattern[0][k] for k in skeleton_indices]), tuple(skeleton_edges))
                elif len(pattern) == 3:
                    pattern = (tuple([pattern[0][k] for k in skeleton_indices]), tuple(skeleton_edges))
                    pattern = pattern + (construct_igraph(pattern[0], pattern[1]), )
        else:
            raise ValueError
    elif isinstance(pattern, ig.Graph):
        pos_tags = list(pattern.vs["label"])
        skeleton_indices = []
        for idx, x in enumerate(pos_tags):
            if idx in template_indices or (x not in OPTIONAL_POS_TAGS and x not in IGNORE_POS_TAGS):
                skeleton_indices.append(idx)
        if len(skeleton_indices) < len(pattern[0]):
            n = len(pattern[0])
            indices_reverse = [-1] * n
            for i, j in enumerate(skeleton_indices):
                indices_reverse[j] = i
            skeleton_edges = []
            for e in pattern.es:
                u, v = e.tuple
                u, v = indices_reverse[u], indices_reverse[v]
                if u != -1 and v != -1:
                    skeleton_edges.append((u, e["label"], v))
            pattern = construct_igraph([pos_tags[idx] for idx in skeleton_indices], skeleton_edges)
    return pattern


def visualize_pattern(pattern):
    if isinstance(pattern, ig.Graph):
        pos_tags = []
        ctr = Counter()
        for pos in pattern.vs["label"]:
            pos_tags.append(pos + str(ctr[pos] + 1))
            ctr[pos] += 1
        pos_lens = [(len(pos) // 4 + 1) * 4 for pos in pos_tags]

        serialization = ["\t\t".join(pos_tags)]

        return "\n".join(serialization)

    elif isinstance(pattern, (tuple, list)) and len(pattern) == 3:
        return visualize_pattern(pattern[2])
    elif isinstance(pattern, (tuple, list)) and len(pattern) == 2:
        if isinstance(pattern[0], (tuple, list)):
            return visualize_pattern(construct_igraph(pattern[0], sorted(pattern[1])))
        elif isinstance(pattern[0], dict):
            indices = []
            for idx, x in pattern[0].items():
                indices.append(idx)
            indices_reverse = {k: -1 for k in pattern[0].keys()}
            indices.sort()
            for i, j in enumerate(indices):
                indices_reverse[j] = i
            edges = []
            for dep in pattern[1]:
                u, elabel, v = dep
                u, v = indices_reverse[u], indices_reverse[v]
                edges.append((u, elabel, v))
            edges.sort()
            return visualize_pattern(construct_igraph([pattern[0][idx] for idx in indices], edges))
        else:
            raise ValueError
    else:
        raise ValueError


def check_match_with_prefix(match, prefix_len):
    reverse = sorted(match)
    i = 1
    while i < prefix_len:
        if reverse[i] - reverse[i - 1] != 1:
            return False
        i += 1
    return True


def check_match_with_must(match, graph):
    match = set(match)
    if isinstance(graph, ig.Graph):
        for eid, dep in enumerate(graph.es):
            if dep.source in match and dep.target not in match and dep["label"] in MUST_POS_TAGS:
                return False
            if dep.source not in match and dep.target in match and dep["label"] in MUST_POS_TAGS:
                return False
    elif isinstance(graph, (tuple, list)) and len(graph) == 3:
        for e in graph[1]:
            if e[0] in match and e[2] not in match and e[1] in MUST_POS_TAGS:
                return False
            if e[0] not in match and e[2] in match and e[1] in MUST_POS_TAGS:
                return False
    elif isinstance(graph, (tuple, list)) and len(graph) == 2:
        for e in graph[1]:
            if e[0] in match and e[2] not in match and e[1] in MUST_POS_TAGS:
                return False
            if e[0] not in match and e[2] in match and e[1] in MUST_POS_TAGS:
                return False
    return True


def check_connect(pattern):
    if isinstance(pattern, ig.Graph):
        if pattern.ecount() < pattern.vcount() - 1:
            return False
        for vid, pos_tag in enumerate(pattern.vs["label"]):
            if pos_tag == "CC" or pos_tag == "WRB" or pos_tag.startswith("WP"):
                in_edges = pattern.incident(vid, mode="in")
                elabels = list(pattern.es[in_edges]["label"])
                if len(
                    in_edges
                ) == 0:  # or ("cc" not in elabels and "dep" not in elabels and "cc:preconj" not in elabels):
                    print("?")
                    return False
        for eid, dep in enumerate(pattern.es["label"]):
            if dep == "parataxis":
                return False
        tmp = pattern.copy()
        tmp.to_undirected()
        return tmp.is_connected()
    elif isinstance(pattern, (tuple, list)) and len(pattern) == 3:
        if len(pattern[1]) < len(pattern[0]) - 1:
            return False
        return check_connect(pattern[2])
    elif isinstance(pattern, (tuple, list)) and len(pattern) == 2:
        if len(pattern[1]) < len(pattern[0]) - 1:
            return False
        if isinstance(pattern[0], (tuple, list)):
            return check_connect(construct_igraph(pattern[0], sorted(pattern[1])))
        elif isinstance(pattern[0], dict):
            indices = []
            for idx, x in pattern[0].items():
                indices.append(idx)
            indices_reverse = {k: -1 for k in pattern[0].keys()}
            indices.sort()
            for i, j in enumerate(indices):
                indices_reverse[j] = i
            edges = []
            for dep in pattern[1]:
                u, elabel, v = dep
                u, v = indices_reverse[u], indices_reverse[v]
                edges.append((u, elabel, v))
            edges.sort()
            return check_connect(construct_igraph([pattern[0][idx] for idx in indices], edges))
        else:
            raise ValueError
    else:
        raise ValueError


pattern_retriever = PatternRetriever()


def construct_pattern(pattern_g, parsed_g, match):
    inv_match = np.argsort(match).tolist()
    new_match = np.argsort(inv_match).tolist()
    pos_tags = pattern_g.vs["label"]
    new_pos_tags = [pos_tags[i] for i in inv_match]
    dep_rels = pattern_g.es["label"]
    new_deps = []
    for rel, (src, dst) in zip(dep_rels, pattern_g.get_edgelist()):
        new_deps.append((new_match[src], rel, new_match[dst]))
    new_deps.sort(key=lambda x: (x[0], x[2]))
    return (tuple(new_pos_tags), tuple(new_deps))


def write_patterns(patterns, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for pattern, cnt in patterns.most_common():
            f.write("t # %d\n" % (cnt))
            for v, p in enumerate(pattern[0]):
                f.write("v %d %s\n" % (v, p))
            for dep in pattern[1]:
                f.write("e %d %d %s\n" % (dep[0], dep[2], dep[1]))


def construct_igraph(pos_tags, dependencies):
    graph = ig.Graph(directed=True)
    graph.add_vertices(len(pos_tags))
    graph.vs["id"] = list(range(len(pos_tags)))
    graph.vs["label"] = list(pos_tags)
    graph.add_edges([(dep[0], dep[2]) for dep in dependencies])
    graph.es["id"] = list(range(len(dependencies)))
    graph.es["label"] = [dep[1] for dep in dependencies]
    return graph


class Node:
    def __init__(self, idx, parents=None, children=None):
        self.idx = idx
        if parents is None:
            self.parents = list()
        else:
            self.parents = [p for p in parents]  # shallow copy
        if children is None:
            self.children = list()
        else:
            self.children = [c for c in children]  # shallow copy

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    # def __dict__(self):
    #     return {"idx": self.idx, "parent": self.parent, "children": self.children}


def build_hierarchy(patterns):
    if len(patterns) == 0:
        return []
    if isinstance(patterns[0], (tuple, list)):
        if len(patterns[0]) == 2:
            return build_hierarchy([construct_igraph(*pattern) for pattern in patterns])
        elif len(patterns[0]) == 3:
            return build_hierarchy([pattern[2] for pattern in patterns])
    elif isinstance(patterns[0], ig.Graph):
        hierarchy = [Node(idx) for idx in range(len(patterns))]
        pattern_retriever = PatternRetriever()
        # sort in a reversed order
        lens = np.array([pattern.vcount() for pattern in patterns])
        indices = np.argsort(lens)
        sorted_patterns = [patterns[idx] for idx in indices]
        rev_indices = np.argsort(indices)

        i = 0
        j = 0
        while i < len(sorted_patterns):
            m = sorted_patterns[i].vcount()
            n = sorted_patterns[i].ecount()
            j = i + 1
            while j < len(sorted_patterns):
                mm = sorted_patterns[j].vcount()
                if mm > m + 1:
                    break
                nn = sorted_patterns[j].ecount()
                if nn < n:
                    j += 1
                    continue
                subisos = pattern_retriever.get_subisomorphisms(
                    sorted_patterns[j], sorted_patterns[i]
                )  # sorted_patterns[j] contains sorted_patterns[i]
                if len(subisos) > 0:
                    hierarchy[rev_indices[i]].parents.append(rev_indices[j])
                    hierarchy[rev_indices[j]].children.append(rev_indices[i])
                j += 1
            i += 1
        return hierarchy
    else:
        raise ValueError


def print_sent(parsed, highlight_idx):
    highlight_idx = set(highlight_idx)
    optional_idx = set()
    for dep in parsed["dependencies"]:

        if dep[0] in highlight_idx and dep[2] not in highlight_idx and parsed["pos_tags"][dep[2]] in OPTIONAL_POS_TAGS:
            optional_idx.add(dep[2])

    for idx, token in enumerate(parsed["tokens"]):
        if idx in highlight_idx:
            print(colored(token, 'red'), end=" ")
        elif idx in optional_idx:
            print(colored(token, 'blue'), end=" ")
        else:
            print(colored(token, 'green'), end=" ")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir_name", type=str, default="parse")
    parser.add_argument("--pattern_dir_name", type=str, default="pattern")
    parser.add_argument("--relation_type", type=str, default="capableOf")
    parser.add_argument("--additional_tokens", nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    args = parser.parse_args()

    rel = args.relation_type
    len_cum = 0
    template = TEMPLATES[rel]
    template_tokens = template[0].split()
    n = len(template_tokens)
    print(f"{rel} template: {template}")

    # load patterns
    pattern_retriever = PatternRetriever()
    meta_patterns = [
        (p[0], p[1], construct_igraph(p[0], p[1]))
        for p in read_patterns(os.path.join(args.pattern_dir_name, f"{rel}-meta.txt"), fuzzy=True)
    ]
    # pprint(meta_patterns)
    patterns = list()
    rel_len1 = len(meta_patterns[0][0])
    for rel_len2 in args.additional_tokens:
        rel_len = rel_len1 + rel_len2
        if not os.path.exists(os.path.join(args.pattern_dir_name, f"{rel}-{rel_len}.txt")):
            continue
        for pattern in read_patterns(os.path.join(args.pattern_dir_name, f"{rel}-{rel_len}.txt"), fuzzy=True):
            pattern_g = construct_igraph(pattern[0], pattern[1])
            if check_connect(pattern_g):
                for i in range(len(meta_patterns)):
                    match = pattern_retriever.get_subisomorphisms(pattern_g, meta_patterns[i][2])
                    if len(match) > 0:
                        # patterns.append((pattern[0], pattern[1], pattern_g))
                        patterns.append(skeletonize_pattern((pattern[0], pattern[1], pattern_g), match[0]))
                        break

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
    print(f"{len(patterns)} patterns after removing redundancies")
    pattern_hierarchy = build_hierarchy(patterns)

    # load data
    process_files = sorted([filename for filename in iter_files(args.processed_dir_name) if rel in filename])
    lines = []
    for filename in process_files:
        if n == 0:
            with open(os.path.join(args.pattern_dir_name, f"{rel}-meta.txt"), "w") as f:
                pass
            with open(os.path.join(args.pattern_dir_name, f"{rel}.lg"), "w") as f:
                pass
        else:
            with open(filename, "r") as f:
                for line in f:
                    line = json.loads(line)
                    if "dependencies" in line and "pos_tags" in line and "tokens" in line:
                        line["graph"] = construct_igraph(line["pos_tags"], line["dependencies"])
                        lines.append(line)
                while len(lines) and isinstance(lines[-1], str) and lines[-1].strip() == "":
                    lines.pop()
    print(f"{len(lines)} lines loaded from {len(process_files)} files")

    N = len(lines)
    pattern_match_flag = [0] * len(lines)
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
            if line_idx in line_used_indices:
                continue

            if pattern_match_flag[line_idx] & flag != 0:
                current_indices.append(line_idx)
                pattern_ctr[pattern_idx] += 1
            elif pattern_match_flag[line_idx] == 0 and len(line["pos_tags"]) >= len(pattern[0]) and len(
                line["dependencies"]
            ) >= len(pattern[1]):
                parsed_g = line["graph"]  # construct_igraph(line["pos_tags"], line["dependencies"])
                # if parsed_g.vcount() - pattern_g.vcount() > 10:
                # parsed_g = parsed_g.induced_subgraph(list(range(pattern_g.vcount() + 10)), "create_from_scratch")
                subisos = pattern_retriever.get_subisomorphisms(parsed_g, pattern_g)
                for subiso in subisos:
                    if check_match_with_prefix(subiso, rel_len1) and check_match_with_must(subiso, parsed_g):
                        current_indices.append(line_idx)
                        pattern_ctr[pattern_idx] += 1

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

                        break

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

        if pattern_ctr[pattern_idx] < N * 1e-4:  # low frequent
            pattern_ctr[pattern_idx] = 0
            continue

        freq_pattern = None
        parsed_g = construct_igraph(example["pos_tags"], example["dependencies"])
        subisos = pattern_retriever.get_subisomorphisms(parsed_g, pattern_g)
        for subiso in subisos:
            if check_match_with_prefix(subiso, rel_len1) and check_match_with_must(subiso, parsed_g):
                freq_pattern = construct_pattern(pattern_g, parsed_g, subiso)
                if freq_pattern[0][-1] not in FORBIDDED_END_POS_TAGS and check_connect(freq_pattern):
                    break

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
    write_patterns(freq_patterns, os.path.join(args.pattern_dir_name, f"{rel}-freq.txt"))
