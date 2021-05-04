from io import TextIOWrapper
import json
from typing import Tuple, List

import nltk
import sklearn_crfsuite


def _parse_ontonotes_x(tokens: List, pos_tags: List) -> List:
    x = []
    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        data = {"word": token, "pos": pos}

        # Look at previous token
        if i > 0:
            data["-1:word.lower()"] = tokens[i - 1].lower()
            data["-1:pos"] = tokens[i - 1].lower()
        else:
            data["BOS"] = True

        # Look at next token
        if i > len(tokens) - 1:
            data["+1:word.lower()"] = tokens[i + 1].lower()
            data["+1:pos"] = tokens[i + 1].lower()
        else:
            data["EOS"] = True

        x.append(data)

    return x


def _parse_ontonotes_y(sentence: dict) -> List:
    y = ["O" for _ in range(len(sentence["tokens"]))]
    if "ne" in sentence and "parse_error" not in sentence["ne"]:
        # Add named entities
        for ne in sentence["ne"].values():
            for i in ne["tokens"]:
                y[i] = ne["type"]

        # Add I/B tags
        y = [y[i] if y[i] == "O" else ("I-" + y[i] if i > 0 and y[i] == y[i-1] else "B-" + y[i])
             for i in range(len(y))]

    return y


def _parse_ontonotes_dataset(ontonotes_file: TextIOWrapper) -> Tuple[List, List]:
    ontonotes = json.load(ontonotes_file)

    x = []
    y = []
    for data_file in ontonotes.values():
        for sentence in data_file.values():
            x.append(_parse_ontonotes_x(list(sentence["tokens"]), list(sentence["pos"])))
            y.append(_parse_ontonotes_y(sentence))

    return x, y


def train_crf_ner_model(ontonotes_file: TextIOWrapper) -> sklearn_crfsuite.CRF:
    """
    Train a CRF Named Entity Recognition model for DATE, CARDINAL, ORDINAL, and NORP entities, using the given dataset.
    """
    print("Parsing dataset")
    x, y = _parse_ontonotes_dataset(ontonotes_file)
    print("Training CRF")
    crf = sklearn_crfsuite.CRF(max_iterations=10, verbose=True)
    crf.fit(x, y)
    print("Finished training CRF")
    return crf


def predict_named_entities(crf: sklearn_crfsuite.CRF, book_file: TextIOWrapper, output_file: str = None):
    """
    Given the file of a plain text book (from www.gutenberg.org) and a trained CRF model, this extracts the named
    entities from the book, returning it as a JSON.
    If `output_file` is not None, then the JSON is outputted to a file with that name.
    """
    book_contents = book_file.read()

    # Tokenize and extract POS tags
    tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(book_contents)
    pos_tags = nltk.pos_tag(tokens)

    # Parse to CRF data and predict named-entities
    x = _parse_ontonotes_x(tokens, pos_tags)
    pred = crf.predict(x)
    print(pred)
    return pred

