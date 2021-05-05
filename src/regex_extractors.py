import re
import json
from typing import TextIO


def extract_table_of_contents(book_file: TextIO, output_file: str = None) -> dict:
    """
    Given the file of a plain text book (from www.gutenberg.org), this extracts the chapter headings and creates a
    table of contents, returning it as a JSON ({<chapter_number>: <chapter_title>}).
    If `output_file` is not None, then the JSON is outputted to a file with that name.
    """

    """
    Things to consider:
    * If a chapter has a number, but no name, e.g. "Chapter 2.", then this becomes {"2": ""} (empty title string)
    * If the chapter heading is split across multiple lines, then don't include the "\n" in the title, just return the
      text (replace the "\n" with a space (unless space already exists)), 
      e.g. "Chapter 3.\nHey guys how \nAre you today" -> {"3": "Hey guys how Are you today"}
    * Use `strip()` to remove any spaces at the start or end
    * May be best to create multiple regex's, rather than one huge regex
    * Use codex.open to read in text. This guarantees that text has UTF-8 chars in it. 
    * DOTALL regex flag means that '.' will match everything, including newlines (doesn't normally match "\n")
    * There will always be a chapter index
    """
    book_contents = book_file.read()

    num_capture_group = r"\b(\d+|[A-Z-]+)\b"
    compiled_pattern = re.compile(
        rf"\n\n\s*([Bb][Oo]{2}[Kk]|[Vv][Oo][Ll](?:[Uu][Mm][Ee])?|[Pp][Aa][Rr][Tt])\s*\n?"  # Volume/Part/Book
        r"[.:-]?\s*"  # Spacing or delimiter
        rf"{num_capture_group}"  # Book number
        r"|"
        r"(?<![Cc][Oo][Nn][Tt][Ee][Nn][Tt][Ss])\n+\n\n\s*([Cc][Hh][Aa][Pp][Tt][Ee][Rr])?\s*\n?"  # A blank line followed by the word 'chapter'
        r"[.:-]?\s*"  # Spacing or delimiter
        rf"{num_capture_group}"  # Chapter number
        r"[.:-]?\s*"  # Spacing or delimiter
        r"(.*)?\s*\n\n",  # Chapter name followed by some spacing
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

    # Save the file to disk
    if output_file is not None:
        with open(output_file + ".json", "w", encoding="utf-8") as save_file:
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

    speech_marks = "‘“\""  # closing quotes: ’”
    title = r"(?:s|iss|rs|r)\.?"
    text = rf"(?: [Mm]{title}|[^!.?‘“\"])"
    compiled_pattern = re.compile(
        rf"((?:M{title}|[A-Z])"  # Start of the question # TODO - Note: Making this [a-zA-Z] resulted in lower F1 score
        rf"{text}*"  # Any text which doesn't end the sentence
        rf"(?:\?|"  # End of question or...
        rf"[{speech_marks}]({text}+\?)))",  # Speech marks followed by a question
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

    # Save the file to disk
    if output_file is not None:
        with open(output_file + ".txt", "w", encoding="utf-8") as save_file:
            text = "\n".join(questions)
            save_file.write(text + "\n")

    return questions
