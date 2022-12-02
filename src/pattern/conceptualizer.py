import argparse
import json
import time
import pickle
import heapq
import os
from tqdm import tqdm
from collections import defaultdict, Counter
from copy import copy, deepcopy
from itertools import combinations, chain
from object import *
from utils import *


class ProbaseConcept(object):
    """ Copied from https://github.com/ScarletPan/probase-concept
    """
    def __init__(self, data_concept_path=""):
        """
        :param data_concept_path: Probase .txt file path
        :type data_concept_path: str
        """
        self.concept2idx = dict()
        self.idx2concept = dict()
        self.concept_inverted_list = dict()
        self.instance2idx = dict()
        self.idx2instance = dict()
        self.instance_inverted_list = dict()
        if data_concept_path:
            self._load_raw_data(data_concept_path)

    def _load_raw_data(self, data_concept_path):
        st = time.time()
        print("[probase-conceptualize] Loading Probase files...")
        with open(data_concept_path) as f:
            triplet_lines = [line.strip() for line in f]

        print("[probase-conceptualize] Building index...")
        for line in tqdm(triplet_lines):
            concept, instance, freq = line.split('\t')
            if concept not in self.concept2idx:
                self.concept2idx[concept] = len(self.concept2idx)
            concept_idx = self.concept2idx[concept]
            if instance not in self.instance2idx:
                self.instance2idx[instance] = len(self.instance2idx)
            instance_idx = self.instance2idx[instance]
            if concept_idx not in self.concept_inverted_list:
                self.concept_inverted_list[concept_idx] = list()
            self.concept_inverted_list[concept_idx].append((instance_idx, int(freq)))
            if instance_idx not in self.instance_inverted_list:
                self.instance_inverted_list[instance_idx] = list()
            self.instance_inverted_list[instance_idx].append((concept_idx, int(freq)))

        self.idx2concept = {val: key for key, val in self.concept2idx.items()}
        self.idx2instance = {val: key for key, val in self.instance2idx.items()}
        print("[probase-conceptualize] Loading data finished in {:.2f} s".format(time.time() - st))

    def conceptualize(self, instance, score_method="likelihood"):
        """ Conceptualize the given instance
        :param instance:  the given instance
        :type instance: str
        :param score_method: the method to compute sscores ("likelihood" or "pmi")
        :type score_method: str
        :return: a list of (concept, score)
        :rtype: List[Tuple[ProbaseConcept, float]]
        """

        if instance not in self.instance2idx:
            return []
        instance_idx = self.instance2idx[instance]
        instance_freq = self.get_instance_freq(instance_idx)
        concept_list = self.instance_inverted_list[instance_idx]
        rst_list = list()
        for concept_idx, co_occurrence in concept_list:
            if score_method == "pmi":
                score = co_occurrence / self.get_concept_freq(concept_idx) / instance_freq
            elif score_method == "likelihood":
                score = co_occurrence / instance_freq
            else:
                raise NotImplementedError
            rst_list.append((self.idx2concept[concept_idx], score))
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def instantiate(self, concept):
        """ Retrieve all instances of a concept
        :param concept: the given concept
        :type concept: str
        :return: a list of instances
        :rtype: List[Tuple[str, float]]
        """

        if concept not in self.concept2idx:
            return []
        concept_idx = self.concept2idx[concept]
        rst_list = [(self.idx2instance[idx], freq) for idx, freq in self.concept_inverted_list[concept_idx]]
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def get_concept_chain(self, instance, max_chain_length=5):
        """ Conceptualize the given instance in a chain
        :param instance: the given instance
        :type instance: str
        :param max_chain_length: the maximum length of the chain
        :type max_chain_length: int (default = 5)
        :return: a chain that contains concepts
        :rtype: List[str]
        """

        if instance in self.concept2idx:
            chain = [instance]
        else:
            chain = list()
        tmp_instance = instance
        while True:
            concepts = self.conceptualize(tmp_instance, score_method="likelihood")
            if concepts:
                chain.append(concepts[0][0])
            else:
                break
            if len(chain) >= max_chain_length:
                break
            tmp_instance = chain[-1]
        if chain and chain[0] != instance:
            return [instance] + chain
        else:
            return chain

    def get_concept_freq(self, concept):
        """ Get the frequency of a concept
        :param concept: the given concept
        :type concept: str
        :return: the corresponding frequency
        :rtype: float
        """

        if isinstance(concept, str):
            if concept not in self.concept2idx:
                return 0
            concept = self.concept2idx[concept]
        elif isinstance(concept, int):
            if concept not in self.idx2concept:
                return 0
        return sum([t[1] for t in self.concept_inverted_list[concept]])

    def get_instance_freq(self, instance):
        """ Get the frequency of an instance
        :param instance: the given instance
        :type instance: str
        :return: the corresponding frequency
        :rtype: float
        """

        if isinstance(instance, str):
            if instance not in self.instance2idx:
                return 0
            instance = self.instance2idx[instance]
        elif isinstance(instance, int):
            if instance not in self.idx2instance:
                return 0
        return sum([t[1] for t in self.instance_inverted_list[instance]])

    def save(self, file_name):
        """
        :param file_name: the file name to save the probase concepts
        :type file_name: str
        """

        with open(file_name, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, file_name):
        """
        :param file_name: the file name to load the probase concepts
        :type file_name: str
        """

        with open(file_name, "rb") as f:
            tmp_dict = pickle.load(f)
        for key, val in tmp_dict.items():
            self.__setattr__(key, val)

    @property
    def concept_size(self):
        return len(self.concept2idx)

    @property
    def instance_size(self):
        return len(self.instance2idx)


class BaseConceptualizer(object):
    """ Base ASER eventuality conceptualizer to conceptualize eventualities
    """
    def __init__(self):
        pass

    def close(self):
        """ Close the ASER Conceptualizer safely
        """
        pass

    def conceptualize(self, eventuality):
        """ Conceptualize an eventuality
        :param eventuality: an eventuality
        :type eventuality: Eventuality
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[ASERConcept, float]]
        """

        raise NotImplementedError

    def conceptualize_from_text(self, words, ners=None):
        """ Conceptualize an eventuality
        :param words: a word list
        :type words: List[str]
        :param ners: a ner list
        :type ners: List[str]
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[ASERConcept, float]]
        """

        raise NotImplementedError


class SeedRuleConceptualizer(BaseConceptualizer):
    """ eventuality conceptualizer based on rules and NERs
    """
    def __init__(self, **kw):
        super().__init__()
        self.selected_ners = frozenset(
            [
                "TIME", "DATE", "DURATION", "MONEY", "PERCENT", "NUMBER", "COUNTRY", "STATE_OR_PROVINCE", "CITY",
                "NATIONALITY", "PERSON", "RELIGION", "URL"
            ]
        )
        self.seed_concepts = frozenset([self._render_ner(ner) for ner in self.selected_ners])

        self.person_pronoun_set = frozenset(
            ["he", "she", "i", "him", "her", "me", "woman", "man", "boy", "girl", "you", "we", "they"]
        )
        self.pronouns = self.person_pronoun_set | frozenset(['it'])

    def conceptualize(self, eventuality):
        """ Conceptualization based on rules and NERs given an eventuality
        :param eventuality: an eventuality
        :type eventuality: Eventuality
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[ASERConcept, float]]
        """

        concept_strs = self.conceptualize_from_text(eventuality.phrases, eventuality.phrases_ners)
        return [(" ".join(concept_strs), 1.0)]

    def conceptualize_from_text(self, words, ners):
        """ Conceptualization based on rules and NERs given a word list an a ner list
        :param words: a word list
        :type words: List[str]
        :param ners: a ner list
        :type ners: List[str]
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[ASERConcept, float]]
        """

        output_words = list()
        ners_dict = {ner: dict() for ner in self.selected_ners}
        for word, ner in zip(words, ners):
            if ner in self.selected_ners:
                if word not in ners_dict[ner]:
                    ners_dict[ner][word] = len(ners_dict[ner])
                output_words.append(self._render_ner(ner) + "%d" % ners_dict[ner][word])
            elif word in self.person_pronoun_set:
                if word not in ners_dict["PERSON"]:
                    ners_dict["PERSON"][word] = len(ners_dict["PERSON"])
                output_words.append(self._render_ner("PERSON") + "%d" % ners_dict["PERSON"][word])
            else:
                output_words.append(word)
        return output_words

    def is_seed_concept(self, word):
        return word in self.seed_concepts

    def is_pronoun(self, word):
        return word in self.pronouns

    def _render_ner(self, ner):
        return "__" + ner + "__"


class ProbaseConceptualizer(BaseConceptualizer):
    """ eventuality conceptualizer based on Probase and NERs
    """
    def __init__(self, probase_path=None, topK=None):
        super().__init__()
        self.seed_conceptualizer = SeedRuleConceptualizer()
        self.probase = ProbaseConcept(probase_path)
        self.topK = topK

    def close(self):
        """ Close the ASER Conceptualizer safely
        """
        del self.probase
        self.probase = None

    def conceptualize(self, eventuality, start_index=0):
        """ Conceptualization use probase given an eventuality
        :param eventuality: an eventuality
        :type eventuality:  Eventuality
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[ASERConcept, float]]
        """
        if not isinstance(eventuality, Eventuality):
            eventuality = Eventuality().from_dict(eventuality)

        # word conceptualization
        if start_index == 0:
            concept_after_seed_rule = self.seed_conceptualizer.conceptualize_from_text(
                eventuality.words, eventuality.ners
            )
            concept_strs = self._get_probase_concepts(concept_after_seed_rule, eventuality.pos_tags)
        else:
            concept_after_seed_rule = self.seed_conceptualizer.conceptualize_from_text(
                ["UNK"] * start_index + eventuality.words[start_index:],
                ["O"] * start_index + eventuality.ners[start_index:]
            )
            concept_after_seed_rule = concept_after_seed_rule.__class__(eventuality.words[:start_index]
                                                                       ) + concept_after_seed_rule[start_index:]
            concept_strs = self._get_probase_concepts(
                concept_after_seed_rule, ["FW"] * start_index + eventuality.pos_tags[start_index:]
            )

        if len(eventuality.phrases) != len(eventuality.words):
            concept_strs1 = concept_strs if concept_strs else []
            for idx, indices in enumerate(eventuality._phrase_segment_indices):
                if start_index in indices:
                    start_index = idx
                    break
            if start_index == 0:
                concept_after_seed_rule2 = self.seed_conceptualizer.conceptualize_from_text(
                    eventuality.phrases, eventuality.phrases_ners
                )
                concept_strs2 = self._get_probase_concepts(concept_after_seed_rule2, eventuality.pos_tags)
            else:
                concept_after_seed_rule2 = self.seed_conceptualizer.conceptualize_from_text(
                    ["UNK"] * start_index + eventuality.phrases[start_index:],
                    ["O"] * start_index + eventuality.phrases_ners[start_index:]
                )
                concept_after_seed_rule2 = concept_after_seed_rule2.__class__(eventuality.phrases[:start_index]
                                                                             ) + concept_after_seed_rule2[start_index:]
                concept_strs2 = self._get_probase_concepts(
                    concept_after_seed_rule2, ["FW"] * start_index + eventuality.pos_tags[start_index:]
                )

            max_len = self.topK**self.topK
            used_concepts = set()
            concept_strs = []
            ptr1, ptr2, l1, l2 = 0, 0, len(concept_strs1), len(concept_strs2)
            while ptr1 < l1 and ptr2 < l2 and len(used_concepts) < max_len:
                if concept_strs1[ptr1][1] > concept_strs2[ptr2][1]:
                    concept_str = " ".join(concept_strs1[ptr1][0])
                    if concept_str not in used_concepts:
                        used_concepts.add(concept_str)
                        concept_strs.append(concept_strs1[ptr1])
                    ptr1 += 1
                else:
                    concept_str = " ".join(concept_strs2[ptr2][0])
                    if concept_str not in used_concepts:
                        used_concepts.add(concept_str)
                        concept_strs.append(concept_strs2[ptr2])
                    ptr2 += 1
            while ptr1 < l1 and len(used_concepts) < max_len:
                concept_str = " ".join(concept_strs1[ptr1][0])
                if concept_str not in used_concepts:
                    used_concepts.add(concept_str)
                    concept_strs.append(concept_strs1[ptr1])
                ptr1 += 1
            while ptr2 < l2 and len(used_concepts) < max_len:
                concept_str = " ".join(concept_strs2[ptr2][0])
                if concept_str not in used_concepts:
                    used_concepts.add(concept_str)
                    concept_strs.append(concept_strs2[ptr2])
                ptr2 += 1

        if not concept_strs and concept_after_seed_rule != " ".join(eventuality.words):
            concept_strs = [(concept_after_seed_rule, 1.0)]

        concept_score_pairs = [
            (ASERConcept(words=concept_str, instances=list()), score) for concept_str, score in concept_strs
        ]
        return concept_score_pairs

    def conceptualize_from_text(self, words, ners, pos_tags, dependencies, start_index=0):
        """ Conceptualization use probase given an eventuality
        :param words: a word list
        :type words: List[str]
        :param ners: a ner list
        :type ners: List[str]
        :param dependencies: the input dependencies
        :type dependencies: List[Tuple[int, str, int]]
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[ASERConcept, float]]
        """

        # word conceptualization
        if start_index == 0:
            concept_after_seed_rule = self.seed_conceptualizer.conceptualize_from_text(words, ners)
            concept_strs = self._get_probase_concepts(concept_after_seed_rule, pos_tags)
        else:
            concept_after_seed_rule = self.seed_conceptualizer.conceptualize_from_text(
                ["UNK"] * start_index + words[start_index:], ["O"] * start_index + ners[start_index:]
            )
            concept_after_seed_rule = concept_after_seed_rule.__class__(words[:start_index]
                                                                       ) + concept_after_seed_rule[start_index:]
            concept_strs = self._get_probase_concepts(
                concept_after_seed_rule, ["FW"] * start_index + pos_tags[start_index:]
            )

        # phrase conceptualization
        phrase_segment_indices = self._dep_compound_segment(words, dependencies)
        phrase_words = list()
        phrase_ners = list()
        phrase_pos_tags = list()
        for _range in phrase_segment_indices:
            st = min(_range)
            end = max(_range) + 1
            if start_index in _range:
                start_index = len(phrase_words)
            phrase_words.append(" ".join(words[st:end]))

            if isinstance(ners[_range[0]], str):
                ner = ners[_range[0]]
            else:
                for x in ners[_range[0]].most_common():
                    if x[0] != "O":
                        ner = x[0]
                        break
            phrase_ners.append(ner)
            phrase_pos_tags.append(pos_tags[_range[0]])

        if len(phrase_words) != len(words):
            concept_strs1 = concept_strs if concept_strs else []

            if start_index == 0:
                concept_after_seed_rule2 = self.seed_conceptualizer.conceptualize_from_text(phrase_words, phrase_ners)
                concept_strs2 = self._get_probase_concepts(concept_after_seed_rule2, phrase_pos_tags)
            else:
                concept_after_seed_rule2 = self.seed_conceptualizer.conceptualize_from_text(
                    ["UNK"] * start_index + phrase_words[start_index:], ["O"] * start_index + phrase_ners[start_index:]
                )
                concept_after_seed_rule2 = concept_after_seed_rule2.__class__(phrase_words[:start_index]
                                                                             ) + concept_after_seed_rule2[start_index:]
                concept_strs2 = self._get_probase_concepts(
                    concept_after_seed_rule2, ["FW"] * start_index + phrase_pos_tags[start_index:]
                )

            max_len = self.topK**self.topK
            used_concepts = set()
            concept_strs = []
            ptr1, ptr2, l1, l2 = 0, 0, len(concept_strs1), len(concept_strs2)
            while ptr1 < l1 and ptr2 < l2 and len(used_concepts) < max_len:
                if concept_strs1[ptr1][1] > concept_strs2[ptr2][1]:
                    concept_str = " ".join(concept_strs1[ptr1][0])
                    if concept_str not in used_concepts:
                        used_concepts.add(concept_str)
                        concept_strs.append(concept_strs1[ptr1])
                    ptr1 += 1
                else:
                    concept_str = " ".join(concept_strs2[ptr2][0])
                    if concept_str not in used_concepts:
                        used_concepts.add(concept_str)
                        concept_strs.append(concept_strs2[ptr2])
                    ptr2 += 1
            while ptr1 < l1 and len(used_concepts) < max_len:
                concept_str = " ".join(concept_strs1[ptr1][0])
                if concept_str not in used_concepts:
                    used_concepts.add(concept_str)
                    concept_strs.append(concept_strs1[ptr1])
                ptr1 += 1
            while ptr2 < l2 and len(used_concepts) < max_len:
                concept_str = " ".join(concept_strs2[ptr2][0])
                if concept_str not in used_concepts:
                    used_concepts.add(concept_str)
                    concept_strs.append(concept_strs2[ptr2])
                ptr2 += 1

        if not concept_strs and concept_after_seed_rule != " ".join(words):
            concept_strs = [(concept_after_seed_rule, 1.0)]

        concept_score_pairs = [
            (ASERConcept(words=concept_str, instances=list()), score) for concept_str, score in concept_strs
        ]
        return concept_score_pairs

    def _get_probase_concepts(self, words, pos_tags):
        word2indices = defaultdict(list)
        for idx, word in enumerate(words):
            word2indices[word].append(idx)

        word2concepts = dict()
        for i in range(len(pos_tags)):
            if i >= len(words):
                break
            word = words[i]
            tag = pos_tags[i]

            if tag.startswith("NN") and (len(word) > 0 and word[0].islower()):
                if self.seed_conceptualizer.is_seed_concept(word) or self.seed_conceptualizer.is_pronoun(word):
                    continue
                elif word not in word2concepts:
                    concepts = self.probase.conceptualize(word, score_method="likelihood")
                    if concepts:
                        concept_set = set()
                        valid_indices = list()
                        for idx, (tmp_concept, score) in enumerate(concepts):
                            tmp = tmp_concept.replace(" ", "-")
                            if tmp not in concept_set:
                                valid_indices.append(idx)
                                concept_set.add(tmp)
                            if len(valid_indices) >= self.topK:
                                break
                        word2concepts[word] = \
                            [(concepts[idx][0].replace(" ", "-"), concepts[idx][1]) for idx in valid_indices]
                    else:
                        continue

        matched_words = list(word2concepts.keys())
        replace_word_tuples = list()
        for i in range(1, len(word2concepts) + 1):
            replace_word_tuples.extend(list(combinations(matched_words, i)))

        output_words_heap = list()
        max_len = self.topK**self.topK
        pre_min_score = 1.0
        min_score = -1.0
        pre_comb_len = 0
        comb_len = 1
        for word_tuples in replace_word_tuples:
            tmp_words_list = [(1.0, words)]
            for word in word_tuples:
                new_tmp_words_list = list()
                # can be further optimized...
                for prob, tmp_words in tmp_words_list:
                    for concept, c_prob in word2concepts[word]:
                        _tmp_words = tmp_words[:]
                        for idx in word2indices[word]:
                            _tmp_words[idx] = concept
                        new_tmp_words_list.append((prob * c_prob, _tmp_words))
                del tmp_words_list
                tmp_words_list = new_tmp_words_list

            for tmp in tmp_words_list:
                if len(output_words_heap) >= max_len:
                    tmp = heapq.heappushpop(output_words_heap, tmp)
                else:
                    heapq.heappush(output_words_heap, tmp)
                if min_score < tmp[0]:
                    min_score = tmp[0]
            comb_len = len(word_tuples)
            if pre_min_score == min_score and pre_comb_len + 1 < comb_len and len(output_words_heap) >= max_len:
                break
            if pre_min_score != min_score:
                pre_min_score = min_score
                pre_comb_len = comb_len

        output_words_list = [heapq.heappop(output_words_heap)[::-1] for i in range(len(output_words_heap))][::-1]
        return output_words_list

    def _dep_compound_segment(self, words, dependencies):
        tmp_compound_tuples = list()
        for governor_idx, dep, dependent_idx in dependencies:
            if dep.startswith("compound"):
                tmp_compound_tuples.append((governor_idx, dependent_idx))

        tmp_compound_tuples = sorted(tmp_compound_tuples)
        compound_tuples = list()
        used_indices = set()
        for i in range(len(tmp_compound_tuples)):
            if i in used_indices:
                continue
            s1 = tmp_compound_tuples[i]
            for j in range(i + 1, len(tmp_compound_tuples)):
                if j in used_indices:
                    continue
                s2 = tmp_compound_tuples[j]
                # s1[0] is the governor
                if s2[0] in set(s1[1:]):
                    s1 = s1 + s2[1:]
                    used_indices.add(j)
                # s2[0] is the governor
                elif s1[0] in set(s2[1:]):
                    s1 = s2 + s1[1:]
                    used_indices.add(j)
                # s1[0] and s2[0] are same
                elif s1[0] == s2[0]:
                    s1 = s1 + s2[1:]
                    used_indices.add(j)
                else:
                    break
            used_indices.add(i)
            # check continuous spans
            sorted_s1 = sorted(s1)
            if sorted_s1[-1] - sorted_s1[0] == len(sorted_s1) - 1:
                compound_tuples.append(s1)
            else:
                s1s = []
                k1 = 0
                k2 = 1
                len_s1 = len(sorted_s1)
                indices = dict(zip(s1, range(len_s1)))
                while k2 < len_s1:
                    if sorted_s1[k2 - 1] + 1 != sorted_s1[k2]:
                        # k1 to k2-1
                        s1s.append(tuple([s1[indices[sorted_s1[k]]] for k in range(k1, k2)]))
                        k1 = k2
                    k2 += 1
                if k1 != k2:
                    s1s.append(tuple([s1[indices[sorted_s1[k]]] for k in range(k1, k2)]))
                compound_tuples.extend(s1s)

        compound_tuples.sort()
        used_indices = set(chain.from_iterable(compound_tuples))

        segment_rst = list()
        word_idx = 0
        compound_idx = 0
        num_words = len(words)
        num_tuples = len(compound_tuples)
        while word_idx < num_words:
            if word_idx not in used_indices:
                segment_rst.append((word_idx, ))
            elif word_idx in used_indices and compound_idx < num_tuples and word_idx == compound_tuples[compound_idx][0]:
                segment_rst.append(compound_tuples[compound_idx])
                compound_idx += 1
            word_idx += 1

        return segment_rst


def conceptualize_eventualities(conceptualizer, eventualities):
    """ Conceptualize eventualities by a conceptualizer
    :param conceptualizer: a conceptualizer
    :type conceptualizer: BaseConceptualizer
    :param eventualities: a list of eventualities
    :type eventualities: List[Eventuality]
    :return: a dictionary from cid to concept, a list of concept-instance pairs, a dictionary from cid to weights
    :rtype: Dict[str, ASERConcept], List[ASERConcept, Eventuality, float], Dict[str, float]
    """

    cid2concept = dict()
    concept_instance_pairs = []
    cid2score = dict()
    for eventuality in tqdm(eventualities):
        results = conceptualizer.conceptualize(eventuality)
        for concept, score in results:
            if concept.cid not in cid2concept:
                cid2concept[concept.cid] = deepcopy(concept)
            concept = cid2concept[concept.cid]
            if (eventuality.eid, eventuality.pattern, score) not in concept.instances:
                concept.instances.append(((eventuality.eid, eventuality.pattern, score)))
                if concept.cid not in cid2score:
                    cid2score[concept.cid] = 0.0
                cid2score[concept.cid] += score * eventuality.frequency
            concept_instance_pairs.append((concept, eventuality, score))
    return cid2concept, concept_instance_pairs, cid2score


def conceptualize_file(input_file, output_file, conceptualizer, start_index=0):
    with open(output_file, "w") as ff:
        with open(input_file, "r") as f:
            for line in f:
                line = json.loads(line)
                if "ners" not in line:
                    line["ners"] = []
                conceptualized_eventualities = []
                for i, eventuality in enumerate(line["eventualities"]):
                    eventuality = Eventuality().from_dict(eventuality)
                    eventuality._phrase_segment_indices = eventuality._phrase_segment()
                    line["eventualities"][i] = eventuality.to_dict(minimum=True)
                    conceptualized_results = conceptualizer.conceptualize(eventuality, start_index)
                    conceptualized_eventualities.append([(x[0].to_dict(), x[1]) for x in conceptualized_results])
                line["conceptualized_eventualities"] = conceptualized_eventualities
                conceptualized_results = conceptualizer.conceptualize_from_text(
                    line["tokens"], line["ners"], line["pos_tags"], line["dependencies"], start_index
                )
                line["conceptualized_text"] = [(x[0].to_dict(), x[1]) for x in conceptualized_results]

                ff.write(json.dumps(line))
                ff.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="extraction/capableOf_elec.jsonl")
    parser.add_argument("--relation_type", type=str, default="capableOf")
    parser.add_argument("--output_file", type=str, default="conceptualization/capableOf_elec.jsonl")
    parser.add_argument("--probase_path", type=str, default="probase/data-concept-instance-relations.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    conceptulizer = ProbaseConceptualizer(args.probase_path)

    conceptualize_file(args.data_file, args.output_file, conceptulizer, len(TEMPLATES[args.relation_type][1]))
