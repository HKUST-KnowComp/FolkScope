import os
import igraph as ig
from collections import defaultdict, Counter
from copy import copy, deepcopy
from itertools import chain
from pattern_retriever import PatternRetriever
from pattern_miner import *
from object import *
from utils import *


class BaseEventualityExtractor(object):
    """ Base ASER eventuality extractor to extract eventualities

    """
    def __init__(self, corenlp_path="", corenlp_port=0, **kw):
        """

        :param corenlp_path: corenlp path, e.g.,stanford-corenlp-4.4.0
        :type corenlp_path: str (default = "")
        :param corenlp_port: corenlp port, e.g., 9000
        :type corenlp_port: int (default = 0)
        :param kw: other parameters
        :type kw: Dict[str, object]
        """

        self.corenlp_path = corenlp_path
        self.corenlp_port = corenlp_port
        self.annotators = kw.get("annotators", list(ANNOTATORS))

        _, self.is_externel_corenlp = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)

    def close(self):
        """ Close the extractor safely
        """

        if not self.is_externel_corenlp:
            corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)
            corenlp_client.stop()

    def __del__(self):
        self.close()

    def parse_text(self, text, annotators=None):
        """ Parse a raw text by corenlp

        :param text: a raw text
        :type text: str
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :return: the parsed result
        :rtype: List[Dict[str, object]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

            [{'dependencies': [(1, 'nmod:poss', 0),
                               (3, 'nsubj', 1),
                               (3, 'aux', 2),
                               (3, 'dobj', 5),
                               (3, 'punct', 6),
                               (5, 'nmod:poss', 4)],
              'lemmas': ['my', 'army', 'will', 'find', 'you', 'boat', '.'],
              'mentions': [],
              'ners': ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
              'parse': '(ROOT (S (NP (PRP$ My) (NN army)) (VP (MD will) (VP (VB find) (NP '
                       '(PRP$ your) (NN boat)))) (. .)))',
              'pos_tags': ['PRP$', 'NN', 'MD', 'VB', 'PRP$', 'NN', '.'],
              'text': 'My army will find your boat.',
              'tokens': ['My', 'army', 'will', 'find', 'your', 'boat', '.']},
             {'dependencies': [(2, 'case', 0),
                               (2, 'det', 1),
                               (6, 'nmod:in', 2),
                               (6, 'punct', 3),
                               (6, 'nsubj', 4),
                               (6, 'cop', 5),
                               (6, 'ccomp', 9),
                               (6, 'punct', 13),
                               (9, 'nsubj', 7),
                               (9, 'aux', 8),
                               (9, 'iobj', 10),
                               (9, 'dobj', 12),
                               (12, 'amod', 11)],
              'lemmas': ['in',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         'be',
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodation',
                         '.'],
              'mentions': [],
              'ners': ['O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O'],
              'parse': '(ROOT (S (PP (IN In) (NP (DT the) (NN meantime))) (, ,) (NP (PRP '
                       "I)) (VP (VBP 'm) (ADJP (JJ sure) (SBAR (S (NP (PRP we)) (VP (MD "
                       'could) (VP (VB find) (NP (PRP you)) (NP (JJ suitable) (NNS '
                       'accommodations)))))))) (. .)))',
              'pos_tags': ['IN',
                           'DT',
                           'NN',
                           ',',
                           'PRP',
                           'VBP',
                           'JJ',
                           'PRP',
                           'MD',
                           'VB',
                           'PRP',
                           'JJ',
                           'NNS',
                           '.'],
              'text': "In the meantime, I'm sure we could find you suitable "
                      'accommodations.',
              'tokens': ['In',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         "'m",
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodations',
                         '.']}]
        """

        if annotators is None:
            annotators = self.annotators

        corenlp_client, _ = get_corenlp_client(
            corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, annotators=annotators
        )
        parsed_result = parse_sentense_with_stanford(text, corenlp_client, self.annotators)
        return parsed_result

    def extract_from_text(self, text, output_format="Eventuality", in_order=True, annotators=None, **kw):
        """ Extract eventualities from a raw text

        :param text: a raw text
        :type text: str
        :param output_format: which format to return, "Eventuality" or "json"
        :type output_format: str (default = "Eventuality")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities
        :rtype: Union[List[List[Eventuality]], List[List[Dict[str, object]]], List[Eventuality], List[Dict[str, object]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

            [[my army will find you boat],
             [i be sure, we could find you suitable accommodation]]
        """

        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_text only supports Eventuality or json.")
        parsed_result = self.parse_text(text, annotators)
        return self.extract_from_parsed_result(parsed_result, output_format, in_order, **kw)

    def extract_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, **kw):
        """ Extract eventualities from the parsed result

        :param parsed_result: the parsed result returned by corenlp
        :type parsed_result: List[Dict[str, object]]
        :param output_format: which format to return, "Eventuality" or "json"
        :type output_format: str (default = "Eventuality")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities
        :rtype: Union[List[List[Eventuality]], List[List[Dict[str, object]]], List[Eventuality], List[Dict[str, object]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            [{'dependencies': [(1, 'nmod:poss', 0),
                               (3, 'nsubj', 1),
                               (3, 'aux', 2),
                               (3, 'dobj', 5),
                               (3, 'punct', 6),
                               (5, 'nmod:poss', 4)],
              'lemmas': ['my', 'army', 'will', 'find', 'you', 'boat', '.'],
              'mentions': [],
              'ners': ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
              'parse': '(ROOT (S (NP (PRP$ My) (NN army)) (VP (MD will) (VP (VB find) (NP '
                       '(PRP$ your) (NN boat)))) (. .)))',
              'pos_tags': ['PRP$', 'NN', 'MD', 'VB', 'PRP$', 'NN', '.'],
              'text': 'My army will find your boat.',
              'tokens': ['My', 'army', 'will', 'find', 'your', 'boat', '.']},
             {'dependencies': [(2, 'case', 0),
                               (2, 'det', 1),
                               (6, 'nmod:in', 2),
                               (6, 'punct', 3),
                               (6, 'nsubj', 4),
                               (6, 'cop', 5),
                               (6, 'ccomp', 9),
                               (6, 'punct', 13),
                               (9, 'nsubj', 7),
                               (9, 'aux', 8),
                               (9, 'iobj', 10),
                               (9, 'dobj', 12),
                               (12, 'amod', 11)],
              'lemmas': ['in',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         'be',
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodation',
                         '.'],
              'mentions': [],
              'ners': ['O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O',
                       'O'],
              'parse': '(ROOT (S (PP (IN In) (NP (DT the) (NN meantime))) (, ,) (NP (PRP '
                       "I)) (VP (VBP 'm) (ADJP (JJ sure) (SBAR (S (NP (PRP we)) (VP (MD "
                       'could) (VP (VB find) (NP (PRP you)) (NP (JJ suitable) (NNS '
                       'accommodations)))))))) (. .)))',
              'pos_tags': ['IN',
                           'DT',
                           'NN',
                           ',',
                           'PRP',
                           'VBP',
                           'JJ',
                           'PRP',
                           'MD',
                           'VB',
                           'PRP',
                           'JJ',
                           'NNS',
                           '.'],
              'text': "In the meantime, I'm sure we could find you suitable "
                      'accommodations.',
              'tokens': ['In',
                         'the',
                         'meantime',
                         ',',
                         'I',
                         "'m",
                         'sure',
                         'we',
                         'could',
                         'find',
                         'you',
                         'suitable',
                         'accommodations',
                         '.']}]

            Output:

            [[my army will find you boat],
             [i be sure, we could find you suitable accommodation]]

        """

        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Eventuality or json.")
        raise NotImplementedError


class PatternMatchEventualityExtractor(BaseEventualityExtractor):
    """ ASER eventuality extractor based on frequent patterns to extract eventualities  (for ASER v3.0)

    """
    def __init__(self, corenlp_path="", corenlp_port=0, **kw):
        """

        :param corenlp_path: corenlp path, e.g.,stanford-corenlp-4.4.0
        :type corenlp_path: str (default = "")
        :param corenlp_port: corenlp port, e.g., 9000
        :type corenlp_port: int (default = 0)
        :param kw: other parameters, e.g., "skip_words" to drop sentences that contain such words
        :type kw: Dict[str, object]
        """
        if "annotators" not in kw:
            kw["annotators"] = list(ANNOTATORS)
        super().__init__(corenlp_path, corenlp_port, **kw)
        self.skip_words = kw.get("skip_words", set())
        if not isinstance(self.skip_words, set):
            self.skip_words = set(self.skip_words)

        self.pattern_retriever = PatternRetriever()
        patterns = list()
        if "patterns" in kw and kw["patterns"] is not None:
            for pattern in kw["patterns"]:
                if isinstance(pattern, ig.Graph):
                    patterns.append(pattern)
                elif isinstance(pattern, (tuple, list)):
                    if len(pattern) == 3:
                        patterns.append(pattern[2])
                    elif len(pattern) == 2:
                        patterns.append(construct_igraph(pattern[0], pattern[1]))
                    else:
                        raise ValueError
                else:
                    raise ValueError

        if "pattern_file" in kw and kw["pattern_file"] is not None:
            for pattern in read_patterns(os.path.join(kw["pattern_file"]), fuzzy=True):
                patterns.append(construct_igraph(pattern[0], pattern[1]))

        # remove redundancies
        patterns.sort(key=lambda x: (x.ecount(), x.vcount(), tuple(x.vs["label"]), tuple(x.es["label"])), reverse=True)
        i = 0
        j = len(patterns) - 1
        duplicate_indices = set()
        for i in range(len(patterns)):
            if i in duplicate_indices:
                continue
            for j in range(i + 1, len(patterns)):
                if j in duplicate_indices:
                    continue
                sub_iso = self.pattern_retriever.get_subisomorphisms(patterns[j], patterns[i])
                if len(sub_iso) > 0:
                    duplicate_indices.add(j)
        patterns = [pattern for idx, pattern in enumerate(patterns) if idx not in duplicate_indices]
        patterns.sort(key=lambda x: (x.ecount(), x.vcount(), tuple(x.vs["label"]), tuple(x.es["label"])), reverse=True)
        self.patterns = patterns
        self.pattern_hierarchy = build_hierarchy(self.patterns)

    def extract_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, **kw):
        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Eventuality or json.")

        if not isinstance(parsed_result, (list, tuple, dict)):
            raise NotImplementedError
        if isinstance(parsed_result, dict):
            is_single_sent = True
            parsed_result = [parsed_result]
        else:
            is_single_sent = False

        para_eventualities = []
        for sent_parsed_result in parsed_result:
            if self.skip_words and set(sent_parsed_result["tokens"]) & self.skip_words:
                continue
            sent_dep_graph = construct_igraph(sent_parsed_result["pos_tags"], sent_parsed_result["dependencies"])

            extracted_eventualities = self._extract_eventualities_from_dependencies(sent_parsed_result, sent_dep_graph)
            extracted_eventualities = self._remove_duplicates(extracted_eventualities)
            # print("-------------")
            para_eventualities.append(extracted_eventualities)

        if in_order:
            para_eventualities = [
                sorted(sent_eventualities, key=lambda e: e.position) for sent_eventualities in para_eventualities
            ]
            if output_format == "json":
                para_eventualities = [
                    [eventuality.encode(encoding=None) for eventuality in sent_eventualities]
                    for sent_eventualities in para_eventualities
                ]
            if is_single_sent:
                return para_eventualities[0]
            else:
                return para_eventualities
        else:
            eid2eventuality = dict()
            for eventuality in chain.from_iterable(para_eventualities):
                eid = eventuality.eid
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = deepcopy(eventuality)
                else:
                    eid2eventuality[eid].update(eventuality)
            if output_format == "Eventuality":
                eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
            elif output_format == "json":
                eventualities = sorted(
                    [eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()],
                    key=lambda e: e["eid"]
                )
            return eventualities

    def _extract_eventualities_from_dependencies(self, sent_parsed_result, sent_dep_graph):
        # boundary_indices = list()
        # for idx, pos_tag in enumerate(sent_parsed_result["pos_tags"]):
        #     if pos_tag == "WRB" or pos_tag.startswith("WP") or pos_tag == "CC":
        #         boundary_indices.append(idx)

        local_eventualities = list()
        pattern_match_flag = (1 << len(self.patterns))
        for pattern_idx, pattern in enumerate(self.patterns):
            flag = (1 << pattern_idx)
            if pattern_match_flag & flag != 0:
                continue

            subisos = self.pattern_retriever.get_subisomorphisms(sent_dep_graph, pattern)
            for subiso in subisos:
                subiso_set = set(subiso)
                selected_edges = list()
                skeleton_dependencies = list()
                for dep_r in sent_parsed_result["dependencies"]:
                    if dep_r[0] in subiso_set and dep_r[2] in subiso_set:
                        selected_edges.append(dep_r)
                        skeleton_dependencies.append(dep_r)
                    elif dep_r[0] in subiso_set and dep_r[2] not in subiso_set and sent_parsed_result["pos_tags"][
                        dep_r[2]] in OPTIONAL_POS_TAGS:
                        same_clause = True
                        # for idx in boundary_indices:
                        #     if idx in subiso_set:
                        #         continue
                        #     if (dep_r[0] - idx) * (dep_r[2] - idx) < 0:
                        #         same_clause = False
                        #         break
                        if same_clause:
                            selected_edges.append(dep_r)

                local_eventualities.append(
                    Eventuality(
                        pattern=construct_pattern(pattern, sent_dep_graph, subiso),
                        dependencies=selected_edges,
                        skeleton_dependencies=skeleton_dependencies,
                        parsed_result=sent_parsed_result
                    )
                )
                pattern_match_flag |= flag
                for child_idx in self.pattern_hierarchy[pattern_idx].children:
                    pattern_match_flag |= (1 << child_idx)
        return local_eventualities

    @staticmethod
    def _remove_duplicates(extracted_eventualities):
        # remove redundancies
        i = 0
        j = len(extracted_eventualities) - 1
        duplicate_indices = set()
        for i in range(len(extracted_eventualities)):
            if i in duplicate_indices:
                continue
            for j in range(i + 1, len(extracted_eventualities)):
                if j in duplicate_indices:
                    continue
                if extracted_eventualities[j].eid == extracted_eventualities[i].eid:
                    duplicate_indices.add(j)
                elif len(
                    set(extracted_eventualities[j]._skeleton_indices) -
                    set(extracted_eventualities[i]._skeleton_indices)
                ) == 0:
                    duplicate_indices.add(j)
                elif len(
                    set(extracted_eventualities[i]._skeleton_indices) -
                    set(extracted_eventualities[j]._skeleton_indices)
                ) == 0:
                    duplicate_indices.add(i)
        filtered_eventualities = [
            eventuality for idx, eventuality in enumerate(extracted_eventualities) if idx not in duplicate_indices
        ]
        return filtered_eventualities


if __name__ == "__main__":
    pm_event_extractor = PatternMatchEventualityExtractor(
        corenlp_path="corenlp-4.4.0",
        corenlp_port=9000,
        annotators=list(ANNOTATORS),
        pattern_file="pattern/v0.1.1/freq.txt",
    )