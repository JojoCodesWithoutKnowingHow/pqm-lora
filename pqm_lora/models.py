"""The models the caption service runs, each at a pinned Hugging Face commit.

PQM's LoRA trainer (session L.1) captions on a rented pod with the two models
L.0's spike chose and measured: a booru tagger, because Illustrious speaks tags,
and JoyCaption for the one sentence a character's captions end with.

A revision is a commit of the model's repo, never a branch: a re-run downloads
the same bytes, as PQM pins every companion at a commit.
"""

TAGGER_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
#: 2024-07-28, the repo's head when L.1 pinned it (2026-10-26).
TAGGER_REVISION = "b25b82a03f7282e41aa2f257a52c7583b710bd1c"
TAGGER_FILES = ("model.onnx", "selected_tags.csv")

JOYCAPTION_REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava"
#: 2025-05-16, the repo's head when L.1 pinned it (2026-10-26).
JOYCAPTION_REVISION = "ebf414ea497a020da0f82df3913e5b6cb8e9663a"
JOYCAPTION_PATTERNS = ("*.json", "*.safetensors")

#: The lowest score the service reports. PQM's editor applies its own threshold
#: (0.35, L.0's) to what comes back, so it can be changed without a re-run.
REPORT_MIN_SCORE = 0.2

#: JoyCaption's settings at L.0: greedy, so the same image gives the same sentence.
MAX_NEW_TOKENS = 120


def label(repo: str, revision: str) -> str:
    """``repo@abcdef0``: what a result says made it."""
    return "%s@%s" % (repo, revision[:7])
