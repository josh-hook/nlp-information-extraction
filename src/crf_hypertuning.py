import json

import numpy as np
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite.metrics import flat_f1_score

import ner_extractors as ner


def tune_crf():
    with open("resources/ontonotes_parsed.json", encoding="utf-8") as ontonotes_file:
        ontonotes = json.load(ontonotes_file)
        x, y = ner.parse_ontonotes_dataset(ontonotes)
        print("Parsed dataset")

    crf = sklearn_crfsuite.CRF()
    space = {
        "max_iterations": [100],
        "algorithm": ["lbfgs"],
        "all_possible_transitions": [True, False],
        "c1": np.linspace(0, 1),
        "c2": np.linspace(0, 1),
    }

    print("Beginning search")
    search = RandomizedSearchCV(
        crf, space,
        cv=3,
        verbose=2,
        n_jobs=-1,
        n_iter=50,
        scoring=make_scorer(
            flat_f1_score,
            average='weighted',
            labels=["B-PERSON", "I_PERSON", "B-DATE", "I-DATE", "B-CARDINAL", "I-CARDINAL",
                    "B-ORDINAL", "I-ORDINAL", "B-NORP", "I-NORP"]
        ),
        return_train_score=True,
    )
    search.fit(x, y)
    return search
