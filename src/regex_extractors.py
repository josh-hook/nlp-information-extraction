import json

def extract_table_of_contents(file, output_file: str = None) -> dict:
    """
    Given the file of a plain text book (from www.gutenberg.org), this extracts the chapter headings and creates a
    table of contents, returning it as a JSON ({<chapter_number>: <chapter_title>}).
    If `output_file` is not None, then the JSON is outputted to a file with that name.
    """
    table_of_contents = {}

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
    questions = {}

    # Save the file to disk
    if output_file is not None:
        with open(output_file + ".json", "w", encoding="utf-8") as save_file:
            json_as_str = json.dumps(questions, indent=2)
            save_file.write(json_as_str + "\n")

    return questions
