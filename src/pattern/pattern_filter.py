import argparse
import json
import os
import subprocess
import re
import math
from multiprocessing import Pool
from collections import defaultdict, Counter
from utils import *


parsemis_path = os.path.join(os.path.dirname(__file__), "parsemis/bin")


def mine_frequent_patterns(
    class_path, input_file, output_file, min_node, max_node, min_edge, max_edge, min_freq, max_freq
):
    if not os.path.exists(class_path):
        try:
            pwd = os.path.dirname(__file__)
            os.system("unzip -o {pwd}/parsemis.zip -d {pwd}parsemis".format(pwd=pwd))
        except:
            print("Failed to unzip parsemis.zip")
            exit(1)

    class_path = class_path.replace("\\", "/")
    input_file = input_file.replace("\\", "/")
    output_file = output_file.replace("\\", "/")
    cmd = f"java -Xmx16g -cp {class_path} de.parsemis.Miner --graphFile={input_file} --outputFile={output_file} --algorithm=gspan --threads=4 --minimumEdgeCount={min_edge} --maximumEdgeCount={max_edge} --minimumNodeCount={min_node} --maximumNodeCount={max_node} --minimumFrequency={min_freq} --maximumFrequency={max_freq}"

    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = p.communicate()[0]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_file_name", type=str, default="parse")
    parser.add_argument("--pattern_dir_name", type=str, default="pattern")
    parser.add_argument("--relation_type", type=str, default="capableOf")
    parser.add_argument("--n_extractors", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.pattern_dir_name, exist_ok=True)

    rel = args.relation_type
    len_cum = 0
    template = TEMPLATES[rel]
    template_tokens = template[0].split()
    n = len(template_tokens)
    print(f"{rel} template: {template}")

    process_files = sorted([filename for filename in iter_files(args.processed_file_name) if rel in filename])
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
                        lines.append(line)
                while len(lines) and isinstance(lines[-1], str) and lines[-1].strip() == "":
                    lines.pop()
    print(f"{len(lines)} lines loaded from {len(process_files)} files")

    patterns = Counter()
    for line in lines:
        if len(line["dependencies"]) > 0:
            match = True
            indices = []
            i = 0
            for x in template_tokens:
                i = index_from(line["tokens"], x, i)
                if i == -1:
                    match = False
                    break
                indices.append(i)
            if match:
                # edge list
                indices_inverse = dict(zip(indices, range(n)))
                pos_tags = [line["pos_tags"][i] for i in indices_inverse]
                deps = []
                for dep in line["dependencies"]:
                    if dep[0] in indices_inverse and dep[2] in indices_inverse:
                        deps.append((indices_inverse[dep[0]], dep[1], indices_inverse[dep[2]]))
                deps.sort(key=lambda x: (x[0], x[2]))
                patterns[(tuple(pos_tags), tuple(deps))] += 1
    print(f"{len(patterns)} meta-patterns mined")

    with open(os.path.join(args.pattern_dir_name, f"{rel}-meta.txt"), "w") as f:
        for pattern, cnt in patterns.most_common():
            f.write("t # %d\n" % (cnt))
            for v, p in enumerate(pattern[0]):
                f.write("v %d %s\n" % (v, p))
            for dep in pattern[1]:
                f.write("e %d %d %s\n" % (dep[0], dep[2], dep[1]))

    len_cum = compute_cumulative_function(Counter([len(line["tokens"]) for line in lines]))
    avg = get_cumulative_mean(len_cum)
    print(f"avg length: {avg}")

    patterns = Counter()
    for line in lines:
        if len(line["dependencies"]) > 0:
            # edge list
            pos_tags = line["pos_tags"]
            deps = sorted(map(tuple, line["dependencies"]), key=lambda x: (x[0], x[2]))
            patterns[(tuple(pos_tags), tuple(deps))] += 1
    print(f"{len(patterns)} patterns mined")

    patterns = patterns.most_common()
    print(f"{len(patterns)} candidate patterns mined")

    # multiprocessing
    max_th = 0.97
    results = []
    early_stop = 0
    template = TEMPLATES[rel]
    n = len(template[0].split())
    if n == 0:
        exit(0)
    N = len_cum[-1][1]
    if N == 0:
        exit(0)
    delta = 1
    NN = get_cumulative_leftmost(len_cum, avg)[1]
    if NN == 0:
        exit(0)
    factor = math.log(1 + NN / N)
    delta = 1
    with Pool(args.n_extractors) as pool:
        while n + delta < math.ceil(avg):
            min_node = n + delta
            max_node = min_node
            min_edge = min_node - 1
            max_edge = min_edge + 1
            input_file = os.path.join(args.pattern_dir_name, f"{rel}-{min_node}.lg")
            output_file = os.path.join(args.pattern_dir_name, f"{rel}-{min_node}.txt")
            cum1 = get_cumulative_leftmost(len_cum, min_node - 1)[1]
            cum2 = get_cumulative_leftmost(len_cum, min_node - 1 + delta)[1]

            min_th = math.log(1 + (cum2 - cum1) / N) * 0.1
            if min_th < 1e-6:
                break
            min_freq = int((N - cum1) * min_th)
            if min_freq < N * 1e-4:
                break
            max_freq = int((N - cum1) * max_th)

            with open(input_file, "w") as f:
                offset = 0
                for (pos_tags, deps), cnt in patterns:
                    if cnt < min_freq * 1e-2:
                        break
                    idx = min(len(pos_tags), math.ceil(min_node + delta))
                    vs = "\n".join(["v %d %s" % (v, p) for v, p in enumerate(pos_tags[:idx])])
                    es = "\n".join(
                        ["e %d %d %s" % (dep[0], dep[2], dep[1]) for dep in deps if dep[0] < idx and dep[2] < idx]
                    )
                    for t in range(offset, offset + cnt):
                        f.write("t # %d\n" % (t))
                        f.write(vs)
                        f.write("\n")
                        f.write(es)
                        f.write("\n")
                    offset += cnt
            delta += 1
            if offset == 0:
                continue
            results.append(
                (
                    rel, min_node,
                    pool.apply_async(
                        mine_frequent_patterns,
                        args=(
                            parsemis_path, input_file, output_file, min_node, max_node, min_edge, max_edge, min_freq,
                            max_freq
                        )
                    )
                )
            )
        pool.close()

        for x in results:
            relation_type, min_node, output = x
            output_file = os.path.join(args.pattern_dir_name, f"{rel}-{min_node}.txt")
            if early_stop > 1:
                with open(output_file, "w", encoding="utf-8") as f:
                    pass
            else:
                output = output.get()
                output = output.decode("utf-8")
                print(relation_type, min_node, output)
                if int(re.findall(r"found (\d+) fragments", output)[0]) == 0:
                    early_stop += 1
