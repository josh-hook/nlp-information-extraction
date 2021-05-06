import re
import json
from typing import TextIO


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
