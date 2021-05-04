from collections import defaultdict
from functools import reduce
from operator import itemgetter
from typing import Tuple, List, TextIO, Optional
import json
import re

import nltk
from sklearn_crfsuite import CRF
from sklearn.base import BaseEstimator


def extract_word_info(word: str, prefix: str = "") -> dict:
    return {
        # Textual information
        # prefix + "word.lower": word.lower(),
        prefix + "word.istitle": word.istitle(),
        prefix + "word.islower": word.islower(),
        prefix + "word.isnumeric": word.isnumeric(),

        # Word prefix and suffix
        prefix + "word.prefix": word[:4],
        prefix + "word.suffix": word[-4:],

        # Word shape information
        prefix + "word.len": len(word),
        # Word shape replaces capitals with X, lowercase with x, digits with d
        prefix + "word.shape": re.sub(r"\d", "d", re.sub("[a-z]", "x", re.sub("[A-Z]", "X", word))),
    }


def parse_ontonotes_x(tokens: List, pos_tags: List) -> List:
    x = []
    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        data = {"word": token, "pos": pos, **extract_word_info(token)}

        # Look at previous token
        if i > 0:
            data.update({
                "-1:word": tokens[i - 1],
                "-1:pos": pos_tags[i - 1],
                **extract_word_info(tokens[i - 1], "-1:"),
            })
        else:
            data["BOS"] = True

        # Look at next token
        if i < len(tokens) - 1:
            data.update({
                "+1:word": tokens[i + 1],
                "+1:pos": pos_tags[i + 1],
                **extract_word_info(tokens[i + 1], "+1:"),
            })
        else:
            data["EOS"] = True

        x.append(data)

    return x


def parse_ontonotes_y(sentence: dict) -> List:
    y = ["O" for _ in range(len(sentence["tokens"]))]
    if "ne" in sentence and "parse_error" not in sentence["ne"]:
        # Add named entities
        for ne in sentence["ne"].values():
            for i in ne["tokens"]:
                y[i] = ne["type"]

        # Add I/B tags
        y = [y[i] if y[i] == "O" else ("I-" + y[i] if i > 0 and y[i] == y[i - 1] else "B-" + y[i])
             for i in range(len(y))]

    return y


def parse_ontonotes_dataset(ontonotes: dict) -> Tuple[List, List]:
    x = []
    y = []
    for data_file in ontonotes.values():
        for sentence in data_file.values():
            x.append(parse_ontonotes_x(list(sentence["tokens"]), list(sentence["pos"])))
            y.append(parse_ontonotes_y(sentence))

    return x, y


class NerModel(BaseEstimator):
    def __init__(self, **kwargs):
        self.crf = CRF(**kwargs)

    def fit(self, x: List[List[dict]], y: List[List]):
        """ Fit the CRF model with some parsed OntoNotes data """
        self.crf.fit(x, y)

    @staticmethod
    def _accumulate_tags(acc, x):
        if x[1].startswith("B") or len(acc) < 1:
            acc.append((x[0], x[1][2:]))
        else:
            x_prev = acc[-1]
            acc[-1] = (x_prev[0] + " " + x[0], x_prev[1])

        return acc

    def predict(self, x: List[List[dict]]) -> List[dict]:
        pred = self.crf.predict(x)

        result = []
        for sentence, y_pred in zip(x, pred):
            tokens = [data["word"] for data in sentence]
            sentence_tagged = reduce(
                # Reduce tags by joining B- and I- tags
                self._accumulate_tags,
                filter(
                    # Filter out NER tags not needed
                    lambda a: "NORP" in a[1] or "CARDINAL" in a[1] or "DATE" in a[1] or "PERSON" in a[1],
                    zip(tokens, y_pred),
                ),
                [],
            )

            tags_dict = defaultdict(set)
            for token, tag in sentence_tagged:
                tags_dict[tag].add(token.lower())

            result.append(tags_dict)

        return result


# def train_crf_ner_model(x: Optional[List] = None,
#                         y: Optional[List] = None,
#                         ontonotes_file: Optional[TextIO] = None,
#                         **kwargs) -> sklearn_crfsuite.CRF:
#     """
#     Train a CRF Named Entity Recognition model for DATE, CARDINAL, ORDINAL, and NORP entities, using the given dataset.
#     """
#     print("Parsing dataset")
#     if ontonotes_file is not None:
#         ontonotes = json.load(ontonotes_file)
#         x, y = _parse_ontonotes_dataset(ontonotes)
#     print("Training CRF")
#     crf = sklearn_crfsuite.CRF(verbose=True, **kwargs)
#     crf.fit(x, y)
#     print("Finished training CRF")
#     return crf


# def predict_named_entities(crf: sklearn_crfsuite.CRF, book_file: TextIO):
#     """
#     Given the file of a plain text book (from www.gutenberg.org) and a trained CRF model, this extracts the named
#     entities from the book, returning it as a JSON.
#     If `output_file` is not None, then the JSON is outputted to a file with that name.
#     """
#     book_contents = book_file.read()
#
#     # Tokenize and extract POS tags
#     tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(book_contents)
#     pos_tags = list(map(itemgetter(1), nltk.pos_tag(tokens)))
#
#     # Parse to CRF data and predict named-entities
#     x = _parse_ontonotes_x(tokens, pos_tags)
#     pred = crf.predict([x])[0]
#
#     def accumulate_tags(acc, x):
#         if x[1].startswith("B") or len(acc) < 1:
#             acc.append((x[0], x[1][2:]))
#         else:
#             x_prev = acc[-1]
#             acc[-1] = (x_prev[0] + " " + x[0], x_prev[1])
#
#         return acc
#
#     x_tagged = reduce(
#         # Reduce tags by joining B- and I- tags
#         accumulate_tags,
#         filter(
#             # Filter out NER tags not needed
#             lambda a: "NORP" in a[1] or "CARDINAL" in a[1] or "DATE" in a[1] or "PERSON" in a[1],
#             zip(tokens, pred),
#         ),
#         [],
#     )
#
#     tags_dict = defaultdict(set)
#     for token, tag in x_tagged:
#         tags_dict[tag].add(token.lower())
#
#     return tags_dict

