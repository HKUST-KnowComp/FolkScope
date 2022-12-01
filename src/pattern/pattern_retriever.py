import igraph as ig
import numpy as np
from collections import Counter, defaultdict

DUMMY_LABEL = "NONE"


def retrieve_multiple_edges(graph, source=-1, target=-1):
    if source != -1:
        e = graph.incident(source, mode=ig.OUT)
        if target != -1:
            e = set(e).intersection(graph.incident(target, mode=ig.IN))
        return ig.EdgeSeq(graph, e)
    else:
        if target != -1:
            e = graph.incident(target, mode=ig.IN)
        else:
            e = list()
        return ig.EdgeSeq(graph, e)


class PatternRetriever(object):
    dummy_label = DUMMY_LABEL

    def __init__(self):
        pass

    @classmethod
    def node_compat_fn(cls, g1, g2, v1, v2):
        if g1.indegree(v1) < g2.indegree(v2):
            return False
        vl2 = g2.vs[v2]["label"]
        if vl2 == DUMMY_LABEL:
            return True
        vl1 = g1.vs[v1]["label"]

        if vl2 in ["NN", "RB", "JJ"]:
            return vl1.startswith(vl2)
        else:
            return vl1 == vl2

    @classmethod
    def edge_compat_fn(cls, g1, g2, e1, e2):
        edge1 = g1.es[e1]
        edge2 = g2.es[e2]
        if edge1.is_loop() != edge2.is_loop():
            return False
        if edge2["label"] == DUMMY_LABEL:
            return True
        # for multiedges
        edges1 = retrieve_multiple_edges(g1, edge1.source, edge1.target)
        edges2 = retrieve_multiple_edges(g2, edge2.source, edge2.target)
        edge1_labels = []
        for el in edges1["label"]:
            edge1_labels.append(el.split(":")[0])  # we ignore details
        edge1_labels = set(edge1_labels)
        for el in edges2["label"]:
            if el not in edge1_labels:
                return False
        return True

    @classmethod
    def get_vertex_color_vectors(cls, g1, g2, seed_v1=-1, seed_v2=-1):
        N1 = g1.vcount()
        N2 = g2.vcount()
        color_vectors = list()

        color1, color2 = DUMMY_LABEL, DUMMY_LABEL
        if seed_v1 != -1:
            color1 = g1.vs[seed_v1]["label"]
        if seed_v2 != -1:
            color2 = g2.vs[seed_v2]["label"]

        if color1 == DUMMY_LABEL and color2 == DUMMY_LABEL:
            color_vectors.append((None, None))
        elif color1 != DUMMY_LABEL and color2 != DUMMY_LABEL:
            if color1 == color2:
                color1 = np.zeros((N1, ), dtype=np.int64)
                color2 = np.zeros((N2, ), dtype=np.int64)
                color1[seed_v1] = 1
                color2[seed_v2] = 1
                color_vectors.append((color1, color2))
        elif color1 != DUMMY_LABEL:
            seed_label = color1
            color1 = np.zeros((N1, ), dtype=np.int64)
            color1[seed_v1] = 1
            for seed_v2, vertex in enumerate(g2.vs):
                if vertex["label"] == seed_label:
                    color2 = np.zeros((N2, ), dtype=np.int64)
                    color2[seed_v2] = 1
                    color_vectors.append((color1, color2))
        else:  # color2 != DUMMY_LABEL
            seed_label = color2
            color2 = np.zeros((N2, ), dtype=np.int64)
            color2[seed_v2] = 1
            for seed_v1, vertex in enumerate(g1.vs):
                if vertex["label"] == seed_label:
                    color1 = np.zeros((N1, ), dtype=np.int64)
                    color1[seed_v1] = 1
                    color_vectors.append((color1, color2))
        return color_vectors

    @classmethod
    def get_edge_color_vectors(cls, g1, g2, seed_e1=-1, seed_e2=-1):
        E1 = len(g1.es)
        E2 = len(g2.es)
        edge_color_vectors = list()

        color1, color2 = DUMMY_LABEL, DUMMY_LABEL
        if seed_e1 != -1:
            color1 = g1.es[seed_e1]["label"]
        if seed_e2 != -1:
            color2 = g2.es[seed_e2]["label"]

        if color1 == DUMMY_LABEL and color2 == DUMMY_LABEL:
            edge_color_vectors.append((None, None))
        elif color1 != DUMMY_LABEL and color2 != DUMMY_LABEL:
            if color1 == color2 and g1.es[seed_e1].is_loop() == g2.es[seed_e2].is_loop():
                edge_color_vectors.append((color1, color2))
        elif color1 != DUMMY_LABEL:
            seed_label = color1
            is_loop = g1.es[seed_e1].is_loop()
            color1 = [0] * E1
            color1[seed_e1] = 1
            for seed_e2, edge in enumerate(g2.es):
                if edge["label"] == seed_label and is_loop == edge.is_loop():
                    color2 = [0] * E2
                    color2[seed_e2] = 1
                    edge_color_vectors.append((color1, color2))
        else:  # color2 != DUMMY_LABEL:
            seed_label = color2
            is_loop = g2.es[seed_e2].is_loop()
            color2 = [0] * E2
            color2[seed_e2] = 1
            for seed_e1, edge in enumerate(g1.es):
                if edge["label"] == seed_label and is_loop == edge.is_loop():
                    color1 = [0] * E1
                    color1[seed_e1] = 1
                    edge_color_vectors.append((color1, color2))

        return edge_color_vectors

    def check(self, graph, pattern, **kw):
        # valid or not
        if graph.vcount() < pattern.vcount():
            return False
        if graph.ecount() < pattern.ecount():
            return False

        graph_vlabels = Counter()
        for vl in graph.vs["label"]:
            if vl == DUMMY_LABEL:
                continue
            if vl.startswith("NN"):
                vl = "NN"
            elif vl.startswith("RB"):
                vl = "RB"
            elif vl.startswith("JJ"):
                vl = "JJ"
            graph_vlabels[vl] += 1

        pattern_vlabels = Counter()
        for vl in pattern.vs["label"]:
            if vl == DUMMY_LABEL:
                continue
            # if vl.startswith("NN"):
            #     vl = "NN"
            # elif vl.startswith("RB"):
            #     vl = "RB"
            # elif vl.startswith("JJ"):
            #     vl = "JJ"
            pattern_vlabels[vl] += 1
        if len(graph_vlabels) < len(pattern_vlabels):
            return False
        for vertex_label, pv_cnt in pattern_vlabels.most_common():
            diff = graph_vlabels[vertex_label] - pv_cnt
            if diff < 0:
                return False

        graph_elabels = set()
        for el in graph.es["label"]:
            if el == DUMMY_LABEL:
                continue
            el = el.split(":")[0]
            graph_elabels.add(el)
        pattern_elabels = set()
        for el in pattern.es["label"]:
            if el == DUMMY_LABEL:
                continue
            # el = el.split(":")[0]
            pattern_elabels.add(el)
        if len(graph_elabels) < len(pattern_elabels):
            return False

        pattern_esource = defaultdict(Counter)
        pattern_etarget = defaultdict(Counter)
        graph_esource = defaultdict(Counter)
        graph_etarget = defaultdict(Counter)
        for (source, target), edge_label in zip(pattern.get_edgelist(), pattern.es["label"]):
            if edge_label == DUMMY_LABEL:
                continue
            edge_label = edge_label.split(":")[0]
            pattern_esource[edge_label][source] += 1
            pattern_etarget[edge_label][target] += 1
        for (source, target), edge_label in zip(graph.get_edgelist(), graph.es["label"]):
            if edge_label == DUMMY_LABEL:
                continue
            edge_label = edge_label.split(":")[0]
            if edge_label not in pattern_elabels:
                continue
            graph_esource[edge_label][source] += 1
            graph_etarget[edge_label][target] += 1

        for edge_label in pattern_elabels:
            if edge_label in pattern_esource:
                if edge_label not in graph_esource or len(pattern_esource[edge_label]) > len(graph_esource[edge_label]):
                    return False
                pattern_sorted = pattern_esource[edge_label].most_common()
                graph_sorted = graph_esource[edge_label].most_common()
                for i in range(len(pattern_sorted)):
                    if pattern_sorted[i][1] > graph_sorted[i][1]:
                        return False
            if edge_label in pattern_etarget:
                if edge_label not in graph_etarget or len(pattern_etarget[edge_label]) > len(graph_etarget[edge_label]):
                    return False
                pattern_sorted = pattern_etarget[edge_label].most_common()
                graph_sorted = graph_etarget[edge_label].most_common()
                for i in range(len(pattern_sorted)):
                    if pattern_sorted[i][1] > graph_sorted[i][1]:
                        return False
        return True

    def get_subisomorphisms(self, graph, pattern, **kw):
        if not self.check(graph, pattern):
            return list()

        seed_v1 = kw.get("seed_v1", -1)
        seed_v2 = kw.get("seed_v2", -1)
        seed_e1 = kw.get("seed_e1", -1)
        seed_e2 = kw.get("seed_e2", -1)

        vertex_color_vectors = PatternRetriever.get_vertex_color_vectors(
            graph, pattern, seed_v1=seed_v1, seed_v2=seed_v2
        )
        edge_color_vectors = PatternRetriever.get_edge_color_vectors(graph, pattern, seed_e1=seed_e1, seed_e2=seed_e2)

        vertices_in_graph = list()
        if seed_v1 != -1:
            vertices_in_graph.append(seed_v1)
        if seed_e1 != -1:
            vertices_in_graph.extend(graph.es[seed_e1].tuple)
        subisomorphisms = list()  # [(component, mapping), ...]
        for vertex_colors in vertex_color_vectors:
            for edge_colors in edge_color_vectors:
                for subisomorphism in graph.get_subisomorphisms_vf2(
                    pattern,
                    color1=vertex_colors[0],
                    color2=vertex_colors[1],
                    edge_color1=edge_colors[0],
                    edge_color2=edge_colors[1],
                    node_compat_fn=PatternRetriever.node_compat_fn,
                    edge_compat_fn=PatternRetriever.edge_compat_fn
                ):
                    if len(vertices_in_graph) == 0 or all([v in subisomorphism for v in vertices_in_graph]):
                        subisomorphisms.append(subisomorphism)
        return subisomorphisms

    def count_subisomorphisms(self, graph, pattern, **kw):
        if not self.check(graph, pattern):
            return 0

        seed_v1 = kw.get("seed_v1", -1)
        seed_v2 = kw.get("seed_v2", -1)
        seed_e1 = kw.get("seed_e1", -1)
        seed_e2 = kw.get("seed_e2", -1)

        vertex_color_vectors = PatternRetriever.get_vertex_color_vectors(
            graph, pattern, seed_v1=seed_v1, seed_v2=seed_v2
        )
        edge_color_vectors = PatternRetriever.get_edge_color_vectors(graph, pattern, seed_e1=seed_e1, seed_e2=seed_e2)

        vertices_in_graph = list()
        if seed_v1 != -1:
            vertices_in_graph.append(seed_v1)
        if seed_e1 != -1:
            vertices_in_graph.extend(graph.es[seed_e1].tuple)
        if len(vertices_in_graph) == 0:
            counts = 0
            for vertex_colors in vertex_color_vectors:
                for edge_colors in edge_color_vectors:
                    counts += graph.count_subisomorphisms_vf2(
                        pattern,
                        color1=vertex_colors[0],
                        color2=vertex_colors[1],
                        edge_color1=edge_colors[0],
                        edge_color2=edge_colors[1],
                        node_compat_fn=PatternRetriever.node_compat_fn,
                        edge_compat_fn=PatternRetriever.edge_compat_fn
                    )
            return counts
        else:
            counts = 0
            for vertex_colors in vertex_color_vectors:
                for edge_colors in edge_color_vectors:
                    for subisomorphism in graph.get_subisomorphisms_vf2(
                        pattern,
                        color1=vertex_colors[0],
                        color2=vertex_colors[1],
                        edge_color1=edge_colors[0],
                        edge_color2=edge_colors[1],
                        node_compat_fn=PatternRetriever.node_compat_fn,
                        edge_compat_fn=PatternRetriever.edge_compat_fn
                    ):
                        if all([v in subisomorphism for v in vertices_in_graph]):
                            counts += 1
            return counts