import json
import re


def extract_table_of_contents(file, output_file: str = None) -> dict:
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
    book_contents = file.read()

    compiled_pattern = re.compile(
        r"\n\n\s+chapter"  # A blank line followed by the word 'chapter'
        r"\s+(\d+|[A-Z]+)"  # Chapter number
        r"[.:]?\s*"  # Spacing or delimeter
        r"(.*)?\s*\n",  # Chapter name followed by some spacing
        re.IGNORECASE,
    )
    found_chapters = re.findall(compiled_pattern, book_contents)

    table_of_contents = {}
    for num, title in found_chapters:
        table_of_contents[num.strip()] = title.strip()

    # Save the file to disk
    if output_file is not None:
        with open(output_file + ".json", "w", encoding="utf-8") as save_file:
            json_as_str = json.dumps(table_of_contents, indent=2)
            save_file.write(json_as_str + "\n")

    return table_of_contents


def extract_questions(file, output_file: str = None) -> set:
    """
    Given the file of a plain text book (from www.gutenberg.org), this extracts every question in the book, returning
    a set of the questions.
    If `output_file` is not None, then the set of questions is outputted to a plain text file.
    """
    book_contents = file.read()

    speech_marks = "‘“"
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

    # Save the file to disk
    if output_file is not None:
        with open(output_file + ".txt", "w", encoding="utf-8") as save_file:
            text = "\n".join(questions)
            save_file.write(text + "\n")

    return questions
