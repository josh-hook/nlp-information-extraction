import json

from regex_extractors import extract_table_of_contents, extract_questions


def eval_regex_table_of_contents(file, expected: dict):
    prediction = extract_table_of_contents(file)
    if len(prediction) != len(expected):
        return False

    for p, e in zip(prediction.items(), expected.items()):
        if p != e:
            return False

    return True


if __name__ == "__main__":
    with open("../resources/eval_book.txt", "r", encoding="utf-8") as book_file:
        print(extract_table_of_contents(book_file))

        with open("../resources/gold_toc.json", "r", encoding="utf-8") as expected_file:
            print(eval_regex_table_of_contents(book_file, json.load(expected_file)))
