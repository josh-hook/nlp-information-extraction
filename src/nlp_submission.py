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

import nltk
import numpy
import scipy
import sklearn
import sklearn_crfsuite
import sklearn_crfsuite.metrics

warnings.simplefilter(action='ignore', category=FutureWarning)
LOG_FORMAT = '%(levelname) -s %(asctime)s %(message)s'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger.info('logging started')


def exec_ner(file_chapter=None, ontonotes_file=None):
    # INSERT CODE TO TRAIN A CRF NER MODEL TO TAG THE CHAPTER OF TEXT (subtask 3)
    # USING NER MODEL AND REGEX GENERATE A SET OF BOOK CHARACTERS AND FILTERED SET OF NE TAGS (subtask 4)

    # hardcoded output to show exactly what is expected to be serialized (you should change this)
    dictNE = {
        "CARDINAL": [
            "two",
            "three",
            "one"
        ],
        "ORDINAL": [
            "first"
        ],
        "DATE": [
            "saturday",
        ],
        "NORP": [
            "indians"
        ],
        "PERSON": [
            "creakle",
            "mr. creakle",
            "mrs. creakle",
            "miss creakle"
        ]
    }

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

    # hardcoded output to show exactly what is expected to be serialized
    # dictTOC = {
    #     "1": "I AM BORN",
    #     "2": "I OBSERVE",
    #     "3": "I HAVE A CHANGE"
    # }

    with open(file_book, "r", encoding="utf-8") as book_file:
        book_contents = book_file.read()

    num = r"\d+|[A-Z]+"

    compiled_pattern = re.compile(
        rf"\n\n\n\s*(book|vol(?:ume)?|part)\s+({num})"  # Volume/Part/Book
        r"|"
        r"\n\n\n\s*(chapter)\s+"  # A blank line followed by the word 'chapter'
        rf"({num})"  # Chapter number
        r"[.:-]?\s*"  # Spacing or delimiter
        r"(.*)?\s*\n",  # Chapter name followed by some spacing
        re.IGNORECASE,
    )
    sections = re.findall(compiled_pattern, book_contents)

    prefix_stack = []
    prefix_set = set()
    table_of_contents = {}
    for sec, sec_num, chp, chp_num, chp_title in sections:
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

            chapter = prefix + " " + chp_num
            table_of_contents[chapter.strip()] = chp_title.strip()

    # DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

    writeHandle = codecs.open('toc.json', 'w', 'utf-8', errors='replace')
    strJSON = json.dumps(table_of_contents, indent=2)
    writeHandle.write(strJSON + '\n')
    writeHandle.close()


def exec_regex_questions(file_chapter=None):
    # INSERT CODE TO USE REGEX TO LIST ALL QUESTIONS IN THE CHAPTER OF TEXT (subtask 2)

    # hardcoded output to show exactly what is expected to be serialized
    # setQuestions = {
    #     "Traddles?",
    #     "And another shilling or so in biscuits, and another in fruit, eh?",
    #     "Perhaps you’d like to spend a couple of shillings or so, in a bottle of currant wine by and by, up in the "
    #     "bedroom?",
    #     "Has that fellow’--to the man with the wooden leg--‘been here again?",
    # }

    with open(file_chapter, "r", encoding="utf-8") as book_file:
        book_contents = book_file.read()

    speech_marks = "‘“"  # closing quotes: ’”
    not_punctuation = rf"[^!.?{speech_marks}]"
    compiled_pattern = re.compile(
        rf"([a-zA-Z]"  # Start of the question
        rf"{not_punctuation}*"  # Any text which doesn't end the sentence
        rf"(?:\?|"  # End of question or...
        rf"[{speech_marks}]({not_punctuation}+\?)))",  # Speech marks followed by a question
        flags=re.DOTALL,
    )
    found_questions = re.findall(compiled_pattern, book_contents)
    # From 'So she began: “O Mouse, do you know the way out of this pool?' this will extract:
    # ('So she began: “O Mouse, do you know the way out of this pool?',
    #  'O Mouse, do you know the way out of this pool?')

    questions = set()
    for sentence in found_questions:
        for q in sentence:
            if len(q) > 1:
                questions.add(re.sub(r"\s", " ", q))

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

    #
    # subtask 2 >> extract every question from a provided plain text chapter of text
    # Input >> www.gutenberg.org sourced plain text file for a chapter of a book
    # Output >> questions.txt = plain text set of extracted questions. one line per question.
    #

    exec_regex_questions(chapter_file)

    #
    # subtask 3 (NER) >> train NER using ontonotes dataset, then extract DATE, CARDINAL, ORDINAL, NORP entities from a provided chapter of text
    # Input >> www.gutenberg.org sourced plain text file for a chapter of a book
    # Output >> ne.json = { <ne_type> : [ <phrase>, <phrase>, ... ] }
    #
    # subtask 4 (text classifier) >> compile a list of characters from the target chapter
    # Input >> www.gutenberg.org sourced plain text file for a chapter of a book
    # Output >> characters.txt = plain text set of extracted character names. one line per character name.
    #

    exec_ner(chapter_file, ontonotes_file)
