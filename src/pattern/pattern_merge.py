from pattern_miner import *
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern_dir_name", type=str, default="pattern")
    parser.add_argument("--output_file", type=str, default="pattern/freq.txt")
    args = parser.parse_args()

    # load patterns
    pattern_retriever = PatternRetriever()
    patterns = list()
    for rel in TEMPLATES:
        n = len(TEMPLATES[rel][0].split())
        if n == 0:
            continue
        if not os.path.exists(os.path.join(args.pattern_dir_name, f"{rel}-meta.txt")):
            continue
        meta_patterns = [
            (p[0], p[1], construct_igraph(p[0], p[1]))
            for p in read_patterns(os.path.join(args.pattern_dir_name, f"{rel}-meta.txt"), fuzzy=True)
        ]
        # pprint(meta_patterns)
        if len(meta_patterns) == 0:
            continue
        if not os.path.exists(os.path.join(args.pattern_dir_name, f"{rel}-freq.txt")):
            continue
        for pattern, cnt in zip(
            read_patterns(os.path.join(args.pattern_dir_name, f"{rel}-freq.txt"), fuzzy=True, with_frequency=True)
        ):
            pattern_g = construct_igraph(pattern[0], pattern[1])
            for i in range(len(meta_patterns)):
                match = pattern_retriever.get_subisomorphisms(pattern_g, meta_patterns[i][2])
                if len(match) > 0:
                    patterns.append(((pattern[0], pattern[1], pattern_g), cnt))
                    # patterns.append((skeletonize_pattern((pattern[0], pattern[1], pattern_g), match[0]), cnt))
                    break
    print(f"{len(patterns)} patterns before removing redundancies")

    # remove redundancies
    patterns = list(map(list, sorted(patterns, key=lambda x: (len(x[0][0]), x[0][0]), reverse=True)))
    i = 0
    j = len(patterns) - 1
    duplicate_indices = set()
    for i in range(len(patterns)):
        if i in duplicate_indices:
            continue
        for j in range(i + 1, len(patterns)):
            if j in duplicate_indices:
                continue
            sub_iso = pattern_retriever.get_subisomorphisms(patterns[j][0][2], patterns[i][0][2])
            if len(sub_iso) > 0:
                duplicate_indices.add(j)
                patterns[i][1] += patterns[j][1]
    patterns = Counter(
        {
            (pattern[0][0], pattern[0][1]): pattern[1]
            for idx, pattern in enumerate(patterns) if idx not in duplicate_indices
        }
    )
    print(f"{len(patterns)} patterns after removing redundancies")

    write_patterns(patterns, args.output_file)
