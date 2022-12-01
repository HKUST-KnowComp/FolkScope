import numpy as np
import os
import re
import socket
from itertools import chain, combinations
from stanfordnlp.server import CoreNLPClient, TimeoutException

TEMPLATES = {
    "open": ("", ""),
    "relatedTo": ("they both are related to", "PRP DT VBP JJ TO"),
    "isA": ("they both are a type of", "PRP DT VBP DT NN IN"),
    "partOf": ("they both are a part of", "PRP DT VBP DT NN IN"),
    "madeOf": ("they both are made of", "PRP DT VBP VBN IN"),
    "similarTo": ("they both are similar to", "PRP DT VBP JJ TO"),
    "createdBy": ("they are created by", "PRP VBP VBN IN"),
    "hasA": ("they both have", "PRP DT VBP"),
    "propertOf": ("they both have a property of", "PRP DT VBP DT NN IN"),
    "distinctFrom": ("they are distinct from", "PRP VBP JJ IN"),
    "usedFor": ("they are both used for", "PRP VBP DT VBN IN"),
    "can": ("they could both", "PRP MD CC"),
    "capableOf": ("they both are capable of", "PRP DT VBP JJ IN"),
    "definedAs": ("they both are defined as", "PRP DT VBP VBN IN"),
    "symbolOf": ("they both are symbols of", "PRP DT VBP NNS IN"),
    "mannerOf": ("they both are a manner of", "PRP DT VBP DT NN IN"),
    "deriveFrom": ("they are derived from", "PRP VBP VBN IN"),
    "effect": ("the person will", "DT NN MD"),
    "cause": ("the person wants to", "DT NN VBZ TO"),
    "motivatedBy": ("buying them was motivated by", "VBG PRP VBD VBN IN"),
    "causeEffect": ("the person wants his", "DT NN VBZ PRP$")
}

MUST_POS_TAGS = frozenset(["IN", "CC", "TO"])
OPTIONAL_POS_TAGS = frozenset(["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "DT", "PRP$"])
IGNORE_POS_TAGS = frozenset([".", ",", "``", "''", ":", "$", "(", ")", "#", "-LRB-", "-RRB-"])
FORBIDDED_END_POS_TAGS = frozenset(["IN", "CC", "TO"])

ANNOTATORS = ("tokenize", "ssplit", "pos", "lemma", "depparse", "ner")

TYPE_SET = frozenset(["CITY", "ORGANIZATION", "COUNTRY", "STATE_OR_PROVINCE", "LOCATION", "NATIONALITY", "PERSON"])

PRONOUN_SET = frozenset(
    [
        "i", "I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themself", "themselves"
    ]
)

PUNCTUATION_SET = frozenset(list("""!"#&'*+,-..../:;<=>?@[\]^_`|~""") + ["``", "''"])

CLAUSE_SEPARATOR_SET = frozenset(list(".,:;?!~-") + ["..", "...", "--", "---"])

URL_REGEX = re.compile(
    r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))',
    re.IGNORECASE
)

EMPTY_SENT_PARSED_RESULT = {
    "text": ".",
    "dependencies": [],
    "tokens": ["."],
    "lemmas": ["."],
    "pos_tags": ["."],
    "parse": "(ROOT (NP (. .)))",
    "ners": ["O"],
    "mentions": []
}

MAX_LEN = 1024
MAX_ATTEMPT = 10


def is_port_occupied(ip="127.0.0.1", port=80):
    """ Check whether the ip:port is occupied

    :param ip: the ip address
    :type ip: str (default = "127.0.0.1")
    :param port: the port
    :type port: int (default = 80)
    :return: whether is occupied
    :rtype: bool
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False


def get_corenlp_client(corenlp_path="", corenlp_port=0, annotators=None):
    """

    :param corenlp_path: corenlp path, e.g., stanford-corenlp-4.4.0
    :type corenlp_path: str (default = "")
    :param corenlp_port: corenlp port, e.g., 9000
    :type corenlp_port: int (default = 0)
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :return: the corenlp client and whether the client is external
    :rtype: Tuple[stanfordnlp.server.CoreNLPClient, bool]
    """

    if corenlp_port == 0:
        return None, True

    if not annotators:
        annotators = list(ANNOTATORS)

    os.environ["CORENLP_HOME"] = corenlp_path

    if is_port_occupied(port=corenlp_port):
        try:
            corenlp_client = CoreNLPClient(
                annotators=annotators,
                timeout=99999,
                memory='4G',
                endpoint="http://localhost:%d" % corenlp_port,
                start_server=False,
                be_quiet=False
            )
            # corenlp_client.annotate("hello world", annotators=list(annotators), output_format="json")
            return corenlp_client, True
        except BaseException as err:
            raise err
    elif corenlp_path != "":
        print("Starting corenlp client at port {}".format(corenlp_port))
        corenlp_client = CoreNLPClient(
            annotators=annotators,
            timeout=99999,
            memory='4G',
            endpoint="http://localhost:%d" % corenlp_port,
            start_server=True,
            be_quiet=False
        )
        corenlp_client.annotate("hello world", annotators=list(annotators), output_format="json")
        return corenlp_client, False
    else:
        return None, True


def split_sentence_for_parsing(text, corenlp_client, annotators=None, max_len=MAX_LEN):
    """ Split a long sentence (paragraph) into a list of shorter sentences

    :param text: a raw text
    :type text: str
    :param corenlp_client: the given corenlp client
    :type corenlp_client: stanfordnlp.server.CoreNLPClient
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :param max_len: the max length of a paragraph (constituency parsing cannot handle super-long sentences)
    :type max_len: int (default = 1024)
    :return: a list of sentences that satisfy the maximum length requirement
    :rtype: List[str]
    """

    if len(text) <= max_len:
        return [text]

    texts = text.split("\n\n")
    if len(texts) > 1:
        return list(
            chain.from_iterable(
                map(lambda sent: split_sentence_for_parsing(sent, corenlp_client, annotators, max_len), texts)
            )
        )

    texts = text.split("\n")
    if len(texts) > 1:
        return list(
            chain.from_iterable(
                map(lambda sent: split_sentence_for_parsing(sent, corenlp_client, annotators, max_len), texts)
            )
        )

    texts = list()
    temp = corenlp_client.annotate(text, annotators=["ssplit"], output_format='json')['sentences']
    for sent in temp:
        if sent['tokens']:
            char_st = sent['tokens'][0]['characterOffsetBegin']
            char_end = sent['tokens'][-1]['characterOffsetEnd']
        else:
            char_st, char_end = 0, 0
        if char_st == char_end:
            continue
        if char_end - char_st <= max_len:
            texts.append(text[char_st:char_end])
        else:
            texts.extend(re.split(PUNCTUATION_SET, text[char_st:char_end]))
    return texts


def clean_sentence_for_parsing(text):
    """ Clean the raw text

    :param text: a raw text
    :type text: str
    :return: the cleaned text
    :rtype: str
    """

    #  only consider the ascii
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # replace ref
    text = re.sub(r"<ref(.*?)>", "<ref>", text)

    # replace url
    text = re.sub(URL_REGEX, "<url>", text)
    text = re.sub(r"<url>[\(\)\[\]]*<url>", "<url>", text)

    return text.strip()


def parse_sentense_with_stanford(input_sentence, corenlp_client, annotators=None, max_len=MAX_LEN):
    """

    :param input_sentence: a raw sentence
    :type input_sentence: str
    :param corenlp_client: the given corenlp client
    :type corenlp_client: stanfordnlp.server.CoreNLPClient
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :param max_len: the max length of a paragraph (constituency parsing cannot handle super-long sentences)
    :type max_len: int (default = 1024)
    :return: the parsed result
    :rtype: List[Dict[str, object]]
    """

    if not annotators:
        annotators = list(ANNOTATORS)

    parsed_sentences = list()
    raw_texts = list()
    cleaned_sentence = clean_sentence_for_parsing(input_sentence)
    for sentence in split_sentence_for_parsing(cleaned_sentence, corenlp_client, annotators, max_len):
        while True:
            try:
                parsed_sentence = corenlp_client.annotate(sentence, annotators=annotators, output_format="json")
                parsed_sentence = parsed_sentence["sentences"]
                break
            except TimeoutException as e:
                continue
        for sent in parsed_sentence:
            if sent["tokens"]:
                char_st = sent["tokens"][0]["characterOffsetBegin"]
                char_end = sent["tokens"][-1]["characterOffsetEnd"]
            else:
                char_st, char_end = 0, 0
            raw_text = sentence[char_st:char_end]
            raw_texts.append(raw_text)
        parsed_sentences.extend(parsed_sentence)

    parsed_rst_list = list()
    for sent, text in zip(parsed_sentences, raw_texts):
        enhanced_dependency_list = sent["enhancedPlusPlusDependencies"]
        dependencies = set()
        for relation in enhanced_dependency_list:
            if relation["dep"] == "ROOT":
                continue
            governor_pos = relation["governor"]
            dependent_pos = relation["dependent"]
            dependencies.add((governor_pos - 1, relation["dep"], dependent_pos - 1))
        dependencies = list(dependencies)
        dependencies.sort(key=lambda x: (x[0], x[2]))

        x = {
            "text": text,
            "dependencies": dependencies,
            "tokens": [t["word"] for t in sent["tokens"]],
        }
        if "pos" in annotators:
            x["pos_tags"] = [t["pos"] for t in sent["tokens"]]
        if "lemma" in annotators:
            x["lemmas"] = [t["lemma"] for t in sent["tokens"]]
        if "ner" in annotators:
            mentions = []
            for m in sent["entitymentions"]:
                if m["ner"] in TYPE_SET and m["text"].lower().strip() not in PRONOUN_SET:
                    mentions.append(
                        {
                            "start": m["tokenBegin"],
                            "end": m["tokenEnd"],
                            "text": m["text"],
                            "ner": m["ner"],
                            "link": None,
                            "entity": None
                        }
                    )

            x["ners"] = [t["ner"] for t in sent["tokens"]]
            x["mentions"] = mentions
        if "parse" in annotators:
            x["parse"] = re.sub(r"\s+", " ", sent["parse"])

        parsed_rst_list.append(x)
    return parsed_rst_list


def iter_files(path):
    """ Walk through all files located under a root path

    :param path: the directory path
    :type path: str
    :return: all file paths in this directory
    :rtype: List[str]
    """

    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, file_names in os.walk(path):
            for f in file_names:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def indices_from(sequence, x, start_from=0):
    """ Indicies from a specific start point

    :param sequence: a sequence
    :type sequence: List[object]
    :param x: an object to index
    :type x: object
    :param start_from: start point
    :type start_from: int
    :return: indices of the matched objects
    :rtype: List[int]
    """

    indices = []
    for idx in range(start_from, len(sequence)):
        if x == sequence[idx]:
            indices.append(idx)
    return indices


def index_from(sequence, x, start_from=0):
    """ Index from a specific start point

    :param sequence: a sequence
    :type sequence: List[object]
    :param x: an object to index
    :type x: object
    :param start_from: start point
    :type start_from: int
    :return: index of the first matched object
    :rtype: int
    """
    idx = start_from
    while idx < len(sequence):
        if sequence[idx] == x:
            return idx
        idx += 1
    return -1


def compute_cumulative_function(ctr, normalize=False):
    cum = [(0, 0)]
    for x, f in sorted(ctr.items()):
        cum.append((x, f + cum[-1][1]))
    total = cum[-1][-1]
    if normalize and total != 0:
        for i in range(len(cum)):
            cum[i][1] /= total

    return cum


def get_cumulative_mean(cum):
    s = 0
    x = (0, 0)
    for y in cum:
        s += y[0] * (y[1] - x[1])
        x = y

    if s > 0:
        return s / cum[-1][1]
    else:
        return 0


def get_cumulative_leftmost(cum, x):
    # binary search
    i = 0
    j = len(cum)
    while i < j:
        k = (i + j) // 2
        if cum[k][0] < x:
            i = k + 1
        else:
            j = k
    if i < len(cum):
        return cum[i]
    else:
        return cum[-1]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.asarray(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
