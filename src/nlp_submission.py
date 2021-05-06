# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2021
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2021/01/29
# Project : Teaching
#
######################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import codecs
import json
import math
import time
import warnings
import re
import logging
from collections import defaultdict
from functools import reduce
from operator import itemgetter
from typing import TextIO, Optional, Tuple, List, Union
from random import shuffle

from nltk.corpus import gazetteers
import nltk
import numpy
import scipy
import sklearn
import sklearn_crfsuite
import sklearn_crfsuite.metrics
from sklearn_crfsuite import CRF

warnings.simplefilter(action='ignore', category=FutureWarning)
LOG_FORMAT = '%(levelname) -s %(asctime)s %(message)s'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger.info('logging started')


# REGEX
def extract_table_of_contents(book_file: TextIO, output_file: str = None) -> dict:
    """
    Given the file of a plain text book (from www.gutenberg.org), this extracts the chapter headings and creates a
    table of contents, returning it as a JSON ({<chapter_number>: <chapter_title>}).
    If `output_file` is not None, then the JSON is outputted to a file with that name.
    """
    book_contents = book_file.read()

    # Remove newlines from sentences
    book_contents = re.sub(r"(?<=[^\s])[\n\r](?=[^\s])", " ", book_contents)
    book_contents = re.sub(r"(?<=[^\s])[\n\r](?= *[^\s])", " ", book_contents)

    # Remove book header
    book_body = re.search(
        r"C[Oo][Nn][Tt][Ee][Nn][Tt][Ss][ .:\-—_]*\n*"  # Contents title
        r"(?: *(?:\b[A-Za-z-]{3,}\b)? *\b(?:\d+|[A-Z-]+)\b *[.:\-—_]* *[^\n]*\n{1,3})+"  # Each contents entry
        r"(.*)",  # Match book body + footer
        book_contents,
        flags=re.DOTALL,
    )
    if book_body is not None:
        book_contents = "\n\n" + book_body.groups()[0]

    # Remove book footer
    book_body = re.search(
        r"(.*)"  # Match book body
        r"PROJECT GUTENBERG EBOOK",  # Beginning of the book's footer
        book_contents,
        flags=re.DOTALL,
    )
    if book_body is not None:
        book_contents = book_body.groups()[0]

    # Extract ToC
    delimiter = r"[.:\-—_]*\s*"
    compiled_pattern = re.compile(
        # Volume/Part/Book
        rf"\n+\s*([Bb][Oo][Oo][Kk]|[Vv][Oo][Ll](?:[Uu][Mm][Ee])?|[Pp][Aa][Rr][Tt])\s*\n?"  # Volume/Part/Book
        rf"{delimiter}"  # Spacing or delimiter
        r"\b(\d{1,4}|[A-Z-—]+)\b\n"  # Book number
        r"|"

        # Chapter heading WITH the word 'Chapter'
        r"(?:\n{2,}\s*([Cc][Hh][Aa][Pp][Tt][Ee][Rr])\s*"  # A blank line followed by the word 'chapter'
        rf"{delimiter}"  # Spacing or delimiter
        r"\b(\d{1,4}|[IVXMLC]+)\b"  # Chapter number
        rf"[.:\-—_]* *"  # Spacing or delimiter
        r"|"

        # Chapter heading WITHOUT the word 'Chapter'
        r"\n{3,}\s*\b(\d{1,4}|[IVXMLC]+)\b"  # Spacing followed by chapter number
        r"[.:\-—_]+ *)"  # Delimiter

        # Chapter heading text
        r"\n{0,2}(.*)? *\n{2,}",  # Chapter name followed by some spacing
    )
    sections = re.findall(compiled_pattern, book_contents)

    prefix_stack = []
    prefix_set = set()
    table_of_contents = {}
    for sec, sec_num, chp, chp_num, chp_num_no_header, chp_title in sections:
        # Update section
        if len(sec) > 0:
            # Remove old section
            if sec in prefix_set:
                prefix = ""
                while len(prefix_stack) > 0 and sec != prefix:
                    prefix, _ = prefix_stack.pop()
                    prefix_set.remove(prefix)

            # Add new section
            prefix_stack.append((sec, sec_num))
            prefix_set.add(sec)

        # Update chapter
        else:
            # Build prefix
            prefix = " ".join(["(%s %s)" % (s.lower().capitalize(), n) for s, n in prefix_stack])

            chapter = prefix + " " + (chp_num if len(chp) > 0 else chp_num_no_header)
            table_of_contents[chapter.strip()] = chp_title.strip()

    # Save the file to disk
    if output_file is not None:
        with open(output_file + ".json", "w", encoding="utf-8-sig") as save_file:
            json_as_str = json.dumps(table_of_contents, indent=2)
            save_file.write(json_as_str + "\n")

    return table_of_contents


def extract_questions(book_file: TextIO, output_file: str = None) -> set:
    """
    Given the file of a plain text book (from www.gutenberg.org), this extracts every question in the book, returning
    a set of the questions.
    If `output_file` is not None, then the set of questions is outputted to a plain text file.
    """
    book_contents = book_file.read()

    # Replace all whitespace with a single space
    book_contents = re.sub(r"\s+", " ", book_contents)

    # Extract questions
    name = r"(?:[A-Z]\. ?)+[A-Z][a-z]+"
    title = rf"(?<![a-z])(?:(?:[Mm](?:s|iss|rs|r)|[Dd]r|[Ss]t|[Ss]r|[Jj]r|[Pp]rof|[Hh]on|[Rr]ev)\.?|{name})"
    title_upper_case = rf"(?:(?:M(?:s|iss|rs|r)|Dr|St|Sr|Jr|Prof|Hon|Rev)\.?|{name})"
    text = rf"(?:{title} ?|[^!.?‘“\"”])"

    compiled_pattern = re.compile(
        # Start of the question
        rf"((?:{title_upper_case}|"  # Starts with a title (e.g. "Mr. John?")
        rf"(?<=[.!?;][-—])[a-z]|"  # Starts with a hyphen after a sentence (e.g. "What?-what?")
        rf"(?<=[.!?;]\s)[a-z]|"  # Starts with a lowercase character after punctuation (e.g. "Hey! what?")
        # Starts with a lowercase character after punctuation and speech marks (e.g. '"Hey!" what?')
        rf"(?<=[.!?;][‘“\"”]\s)[a-z]|"
        rf"(?<![a-z]\s)[A-Z]|"  # Starts with an uppercase character and is not preceded by a lowercase character
        rf"(?<=[‘“\"])[a-z])"  # Start of speech with a lowercase character

        # Text body
        rf"(?:{text}*\?|"  # Text followed by a question mark (for basic questions)
        # Speech marks surrounding some added context (e.g. in 'This is"-- he exclaimed --"my question?')
        rf"(?:{text}*[’”\"][^A-Za-z!.?‘“\"”]{text}+[‘“\"])+"
        rf"({text}+\?)))",  # Capture the end of the question
        flags=re.DOTALL,
    )
    found_questions = re.findall(compiled_pattern, book_contents)
    # From 'This is"-- he exclaimed --"my question?' this will extract:
    # ('This is"-- he exclaimed --"my question?', 'my question?')

    questions = set()
    for sentence in found_questions:
        for q in sentence:
            if len(q) > 1:
                questions.add(re.sub(r"\s+", " ", q).strip())

    # Save the file to disk
    if output_file is not None:
        with open(output_file + ".txt", "w", encoding="utf-8-sig") as save_file:
            text = "\n".join(questions)
            save_file.write(text + "\n")

    return questions


# NER
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
        prefix + "word.ispunctuation": re.search(r"[.!?,~\-_—:;'\"‘’“”]", word) is not None,
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


# EXEC
def exec_ner(file_chapter=None, ontonotes_file=None):
    # INSERT CODE TO TRAIN A CRF NER MODEL TO TAG THE CHAPTER OF TEXT (subtask 3)
    # USING NER MODEL AND REGEX GENERATE A SET OF BOOK CHARACTERS AND FILTERED SET OF NE TAGS (subtask 4)

    with open(ontonotes_file, encoding="utf-8-sig") as file:
        crf = train_crf_ner_model(ontonotes_file=file, num_sentences=0.8)
        logger.info("Trained CRF")

    with open(chapter_file, encoding="utf-8-sig") as file:
        pred = predict_named_entities(crf, file)

    dictNE = pred
    # DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

    # write out all PERSON entries for character list for subtask 4
    writeHandle = codecs.open('characters.txt', 'w', 'utf-8', errors='replace')
    if 'PERSON' in dictNE:
        for strNE in dictNE['PERSON']:
            writeHandle.write(strNE.strip().lower() + '\n')
    writeHandle.close()

    # FILTER NE dict by types required for subtask 3
    listAllowedTypes = ['DATE', 'CARDINAL', 'ORDINAL', 'NORP']
    listKeys = list(dictNE.keys())
    for strKey in listKeys:
        for nIndex in range(len(dictNE[strKey])):
            dictNE[strKey][nIndex] = dictNE[strKey][nIndex].strip().lower()
        if not strKey in listAllowedTypes:
            del dictNE[strKey]

    # write filtered NE dict
    writeHandle = codecs.open('ne.json', 'w', 'utf-8', errors='replace')
    strJSON = json.dumps(dictNE, indent=2)
    writeHandle.write(strJSON + '\n')
    writeHandle.close()


def exec_regex_toc(file_book=None):
    """
    Subtask 1: Extract chapter headings and create a table of contents, from a plain text book from www.gutenberg.org

    Input: File for a whole plain text book, from www.gutenberg.org
    Output: Create a 'toc.json' file of {<chapter_number_text>: <chapter_title_text>}
    """

    with open(file_book, "r", encoding="utf-8-sig") as book_file:
        table_of_contents = extract_table_of_contents(book_file)

    # DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

    writeHandle = codecs.open('toc.json', 'w', 'utf-8', errors='replace')
    strJSON = json.dumps(table_of_contents, indent=2)
    writeHandle.write(strJSON + '\n')
    writeHandle.close()


def exec_regex_questions(file_chapter=None):
    # INSERT CODE TO USE REGEX TO LIST ALL QUESTIONS IN THE CHAPTER OF TEXT (subtask 2)

    with open(file_chapter, "r", encoding="utf-8-sig") as book_file:
        questions = extract_questions(book_file)

    # DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

    writeHandle = codecs.open('questions.txt', 'w', 'utf-8', errors='replace')
    for strQuestion in questions:
        writeHandle.write(strQuestion + '\n')
    writeHandle.close()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise Exception('missing command line args : ' + repr(sys.argv))
    ontonotes_file = sys.argv[1]
    book_file = sys.argv[2]
    chapter_file = sys.argv[3]

    logger.info('ontonotes = ' + repr(ontonotes_file))
    logger.info('book = ' + repr(book_file))
    logger.info('chapter = ' + repr(chapter_file))

    # DO NOT CHANGE THE CODE IN THIS FUNCTION

    #
    # subtask 1 >> extract chapter headings and create a table of contents from a provided plain text book (from www.gutenberg.org)
    # Input >> www.gutenberg.org sourced plain text file for a whole book
    # Output >> toc.json = { <chapter_number_text> : <chapter_title_text> }
    #

    exec_regex_toc(book_file)
    logger.info("Finished ToC")

    #
    # subtask 2 >> extract every question from a provided plain text chapter of text
    # Input >> www.gutenberg.org sourced plain text file for a chapter of a book
    # Output >> questions.txt = plain text set of extracted questions. one line per question.
    #

    exec_regex_questions(chapter_file)
    logger.info("Finished Questions")

    #
    # subtask 3 (NER) >> train NER using ontonotes dataset, then extract DATE, CARDINAL, ORDINAL, NORP entities
    # from a provided chapter of text
    # Input >> www.gutenberg.org sourced plain text file for a chapter of a book
    # Output >> ne.json = { <ne_type> : [ <phrase>, <phrase>, ... ] }
    #
    # subtask 4 (text classifier) >> compile a list of characters from the target chapter
    # Input >> www.gutenberg.org sourced plain text file for a chapter of a book
    # Output >> characters.txt = plain text set of extracted character names. one line per character name.
    #

    exec_ner(chapter_file, ontonotes_file)
    logger.info("Finished NER")
