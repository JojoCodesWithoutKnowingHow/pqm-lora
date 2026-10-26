"""The tagger's scores turned into tags: standard library only, so it is tested anywhere.

``selected_tags.csv`` lists the model's outputs in order, each with a category:
``0`` general, ``4`` character, ``9`` rating. Names are reported as the model has
them (underscores and all); PQM turns them into caption text.
"""

import csv

GENERAL = "0"
CHARACTER = "4"
RATING = "9"


def read_labels(path):
    """``[(name, category), ...]`` from ``selected_tags.csv``, in output order."""
    with open(path, encoding="utf-8", newline="") as handle:
        return [(row["name"], str(row["category"])) for row in csv.DictReader(handle)]


def select(scores, labels, min_score):
    """``{"general": [[name, score]], "character": [...], "rating": [...]}``.

    General and character tags at or above ``min_score``, most confident first
    (a tie keeps the model's order); every rating, most likely first. Scores are
    rounded to four places, which is all a threshold needs and keeps the JSON
    the same from run to run on one machine.
    """
    if len(scores) != len(labels):
        raise ValueError("%d scores for %d labels" % (len(scores), len(labels)))
    out = {"general": [], "character": [], "rating": []}
    for index, ((name, category), score) in enumerate(zip(labels, scores)):
        score = float(score)
        if category == RATING:
            out["rating"].append((index, name, score))
        elif category in (GENERAL, CHARACTER) and score >= min_score:
            out["general" if category == GENERAL else "character"].append((index, name, score))
    for key in out:
        out[key] = [[name, round(score, 4)]
                    for index, name, score in sorted(out[key], key=lambda r: (-r[2], r[0]))]
    return out
