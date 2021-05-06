from collections import defaultdict
from functools import reduce
from operator import itemgetter
from random import shuffle
from typing import Tuple, List, TextIO, Optional, Union
import json
import re

import nltk
from nltk.corpus import gazetteers
from sklearn_crfsuite import CRF


gazetteer = {word for location in gazetteers.words() for word in location.split()}


def extract_word_info(word: str, pos: str, prefix: str = "") -> dict:
    return {
        # Base information
        prefix + "word": word,
        prefix + "pos": pos,

        # Textual information
        # prefix + "word.lower": word.lower(),
        prefix + "word.istitle": word.istitle(),
        prefix + "word.islower": word.islower(),
        prefix + "word.isnumeric": word.isnumeric(),
        prefix + "word.ispunctuation": re.search(r"[.!?,~\-_—:;'\"‘’“”/\\£$&%]", word) is not None,
        prefix + "word.len": len(word),

        # Word prefix and suffix
        prefix + "word.prefix": word[:3],
        prefix + "word.suffix": word[-3:],

        # Gazetteer
        prefix + "word.ingazetteer": word in gazetteer,
    }


def parse_ontonotes_x(tokens: List, pos_tags: List) -> List:
    x = []
    for i in range(len(tokens)):
        data = extract_word_info(tokens[i], pos_tags[i])
        # Word shape replaces capitals with X, lowercase with x, digits with d
        data["word.shape"] = re.sub(r"\d", "d", re.sub("[a-z]", "x", re.sub("[A-Z]", "X", tokens[i])))

        # Look at previous token
        if i > 0:
            data.update(extract_word_info(tokens[i - 1], pos_tags[i - 1], "-1:"))
        else:
            data["BOS"] = True

        # Look at next token
        if i < len(tokens) - 1:
            data.update(extract_word_info(tokens[i + 1], pos_tags[i + 1], "+1:"))
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


def parse_ontonotes_dataset(ontonotes: dict, num_sentences: Optional[Union[int, float]] = None) -> Tuple[List, List]:
    # Extract sentences and filter out those with incorrect PoS tags
    data = [sentence for data_file in ontonotes.values() for sentence in data_file.values()
            if "XX" not in sentence["pos"] and "VERB" not in sentence["pos"]]
    ontonotes.clear()  # Clear old dict for memory
    shuffle(data)  # Shuffle data

    if num_sentences is not None:
        if isinstance(num_sentences, float):
            data = data[:int(len(data) * num_sentences)]
        else:
            data = data[:num_sentences]

    # Parse OntoNotes sentences
    x = []
    y = []
    # from tqdm import tqdm
    # for sentence in tqdm(data):
    for sentence in data:
        x.append(parse_ontonotes_x(list(sentence["tokens"]), list(sentence["pos"])))
        y.append(parse_ontonotes_y(sentence))

    return x, y


def accumulate_tags(acc, x):
    if x[1].startswith("B") or len(acc) < 1:
        acc.append((x[0], x[1][2:]))
    else:
        x_prev = acc[-1]
        acc[-1] = (x_prev[0] + " " + x[0], x_prev[1])

    return acc


def train_crf_ner_model(x: Optional[List] = None,
                        y: Optional[List] = None,
                        ontonotes_file: Optional[TextIO] = None,
                        num_sentences: Optional[Union[int, float]] = None,
                        **kwargs) -> CRF:
    """
    Train a CRF Named Entity Recognition model for DATE, CARDINAL, ORDINAL, and NORP entities, using the given dataset.
    """
    if ontonotes_file is not None:
        ontonotes = json.load(ontonotes_file)
        x, y = parse_ontonotes_dataset(ontonotes, num_sentences=num_sentences)

    crf = CRF(**{'max_iterations': 40, 'c2': 0.,
                 'c1': 0.4, 'all_possible_transitions': False,
                 'algorithm': 'lbfgs'},
              **kwargs)
    crf.fit(x, y)
    return crf


def update_person(tags_dict: dict, book_contents: str):
    # Clean PERSON tags
    person_set = set()
    for token in tags_dict["PERSON"]:
        extracted_regex_names = re.findall(
            r"((?:\b(?:[A-Z]|Mr|Ms|Mrs|Miss|Dr|St|Sr|Jr|Prof|Hon|Rev)\b\. )*"  # Name prefix
            r"(?<!Miss)(?:\b[A-Z][a-z]{3,}\b ?)+)",  # Name
            token,
        )
        if len(extracted_regex_names) > 0:
            person_set.add(extracted_regex_names[0].lower().strip())

    # Extract names of the form "Mr. John" or "J. John", etc
    person_pattern = re.compile(
        # Name isn't at the start of the sentence
        r"(?<!^)(?<![.!?][\"“‘’] )(?<![.!?] )(?<![\"“‘])(?<!\n)(?<!-)(?<![Tt]he )(?<![Aa] )"
        r"((?:\b(?:[A-Z]|Mr|Ms|Mrs|Miss|Dr|St|Sr|Jr|Prof|Hon|Rev)\b\. )+"  # Name prefix
        r"(?<!Miss)(?:\b[A-Z][a-z]{3,}\b ?)+)"  # Name
    )
    for token in re.findall(person_pattern, book_contents):
        person_set.add(token.lower().strip())

    tags_dict["PERSON"] = person_set


def predict_named_entities(crf: CRF, book_file: TextIO):
    """
    Given the file of a plain text book (from www.gutenberg.org) and a trained CRF model, this extracts the named
    entities from the book, returning it as a JSON.
    If `output_file` is not None, then the JSON is outputted to a file with that name.
    """
    book_contents = book_file.read()
    book_contents = re.sub(r"(?<!\n)\n", " ", book_contents).strip()  # Remove extra spacing and newlines

    # Tokenize and extract POS tags
    tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(book_contents)
    pos_tags = list(map(itemgetter(1), nltk.pos_tag(tokens)))

    # Parse to CRF data and predict named-entities
    x = parse_ontonotes_x(tokens, pos_tags)
    pred = crf.predict([x])[0]

    x_tagged = reduce(
        # Reduce tags by joining B- and I- tags
        accumulate_tags,
        filter(
            # Filter out NER tags not needed
            lambda a: "NORP" in a[1] or "CARDINAL" in a[1] or "DATE" in a[1] or "PERSON" in a[1],
            zip(tokens, pred),
        ),
        [],
    )

    # Put named entities into sets
    tags_dict = defaultdict(set)
    for token, tag in x_tagged:
        if tag == "PERSON":
            tags_dict[tag].add(token.strip())
        else:
            tags_dict[tag].add(token.lower().strip())

    # Update PERSON tags
    update_person(tags_dict, book_contents)

    # Convert sets to lists
    return {k: list(v) for k, v in tags_dict.items()}


# def update_ordinal(tags_dict: dict, book_contents: str):
#     # Clean ORDINAL tags
#     tags_dict["ORDINAL"] = {re.sub("[^a-z]", "", token) for token in tags_dict["ORDINAL"]}
#
#     # Extract ordinals using regex
#     ordinal_pattern = re.compile(
#         r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)"
#     )
#     add_to_set(tags_dict["ORDINAL"], re.findall(ordinal_pattern, book_contents))
#
#
# def update_date(tags_dict: dict, book_contents: str):
#     # Clean DATE tags
#     tags_dict["DATE"] = {re.sub("[^a-z-']", "", token) for token in tags_dict["DATE"]}
#
#     # Extract dates using regex
#     days = r"(?:[Mm]on(?:day)?|[Tt]ue(?:sday)?|[Ww]ed(?:nesday)?|[Tt]hu(?:rsday)?|[Ff]ri(?:day)?|" \
#            r"[Ss]at(?:urday)?|[Ss]un(?:day)?)"
#     months = r"(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une?|[Jj]uly?|" \
#              r"[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)"
#     ordinal_pattern = re.compile(
#         rf"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|(?:\d+(?:st|nd|rd|th)(?! {months})",
#     )
#     add_to_set(tags_dict["ORDINAL"], re.findall(ordinal_pattern, book_contents))