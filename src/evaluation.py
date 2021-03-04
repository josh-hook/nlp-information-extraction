import os
import sys
import json

import pandas as pd

from sklearn.metrics import f1_score

from regex_extractors import extract_table_of_contents, extract_questions


def eval_results(expected, prediction) -> float:
    """ Evaluate predictions against expected, returning the f1 score """
    if len(prediction) != len(expected):
        print("\nPrediction len doesn't match expected len:", len(prediction), "!=", len(expected))
        print("Prediction:", prediction)
        print("Expected:", expected)
        return f1_score(expected[:len(prediction)], prediction, average="macro")

    for p, e in zip(prediction, expected):
        if p != e:
            print("Given:", p, ", but expected:", e)

    return f1_score(expected, prediction, average="macro")


def main():
    """ Read in all resources and evaluate """
    resources_dir_path = sys.argv[1]
    resources = os.listdir(resources_dir_path)

    results_df = pd.DataFrame(columns=("Book", "ToC f1", "Questions f1", "NER f1", "Characters f1"))

    for i, resource in enumerate(resources):
        resource_path = os.path.join(resources_dir_path, resource)

        # Regex table of contents
        eval_path = os.path.join(resource_path, "eval_book.txt")
        gold_path = os.path.join(resource_path, "gold_toc.json")
        with open(eval_path, 'r', encoding="utf-8") as eval_file, open(gold_path, 'r', encoding="utf-8") as gold_file:
            expected = [k + " " + v for k, v in json.load(gold_file).items()]
            prediction = [k + " " + v for k, v in extract_table_of_contents(eval_file).items()]
            toc_f1 = eval_results(expected, prediction)

        # Regex questions
        eval_path = os.path.join(resource_path, "eval_chapter.txt")
        gold_path = os.path.join(resource_path, "gold_questions.txt")
        with open(eval_path, 'r', encoding="utf-8") as eval_file, open(gold_path, 'r', encoding="utf-8") as gold_file:
            expected = gold_file.read().split("\n")[:-1]
            prediction = extract_questions(eval_file)
            questions_f1 = eval_results(list(expected), list(prediction))

        results_df.loc[i] = [resource, toc_f1, questions_f1, 0, 0]

    print("\n" + results_df.sort_values(["ToC f1", "Questions f1", "NER f1", "Characters f1"]).to_markdown())


if __name__ == "__main__":
    """ Run with path to resources directory """
    main()

