"""``python -m pqm_lora.verify``: the caption environment works, before the service starts.

The pod's install step ``verify``: torch sees the GPU, onnxruntime offers CUDA
once torch has loaded, and both models' files are on disk at their commits.
Nothing is loaded onto the GPU: the service does that on its first request.
Exit 0 is set up; anything else names what is missing.
"""

import sys

from . import models


def main():
    problems = []
    import torch

    if not torch.cuda.is_available():
        problems.append("torch %s sees no GPU" % torch.__version__)
    else:
        print("torch %s on %s" % (torch.__version__, torch.cuda.get_device_name(0)))
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        problems.append("onnxruntime %s offers no CUDA (%s)" % (ort.__version__, providers))
    import transformers

    print("onnxruntime %s, transformers %s" % (ort.__version__, transformers.__version__))
    from huggingface_hub import hf_hub_download, snapshot_download

    for name in models.TAGGER_FILES:
        try:
            hf_hub_download(models.TAGGER_REPO, name, revision=models.TAGGER_REVISION,
                            local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - every miss is named
            problems.append("the tagger's %s is not downloaded (%s)" % (name, type(exc).__name__))
    try:
        snapshot_download(models.JOYCAPTION_REPO, revision=models.JOYCAPTION_REVISION,
                          allow_patterns=list(models.JOYCAPTION_PATTERNS), local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        problems.append("JoyCaption is not downloaded (%s)" % type(exc).__name__)
    for line in problems:
        print("NOT READY: " + line, flush=True)
    if not problems:
        print("caption environment ready", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
