"""``python -m pqm_lora.fetch``: both models' files at their pinned commits, into ``HF_HOME``.

The pod's install step ``weights``. Public repos, so no token: PQM never hands a
companion one. Already-downloaded files are checked and kept by huggingface_hub.
"""

import sys
import time

from . import models


def main():
    from huggingface_hub import snapshot_download

    for repo, revision, patterns in (
            (models.TAGGER_REPO, models.TAGGER_REVISION, list(models.TAGGER_FILES)),
            (models.JOYCAPTION_REPO, models.JOYCAPTION_REVISION,
             list(models.JOYCAPTION_PATTERNS))):
        started = time.time()
        where = snapshot_download(repo, revision=revision, allow_patterns=patterns,
                                  max_workers=8)
        print("fetched %s in %.0f s -> %s" % (models.label(repo, revision),
                                               time.time() - started, where), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
