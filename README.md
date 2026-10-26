# pqm-lora

What [Prompt Queue Manager](https://github.com/JojoCodesWithoutKnowingHow)'s LoRA
trainer extension runs on a rented pod. PQM fetches this repo onto the pod at a
pinned commit and runs `setup/pod_setup.sh` part by part; nothing here runs on
the user's machine.

## 0.1.0: the caption service (PQM session L.1)

A dataset's images are captioned on the pod by two models, each at a pinned
Hugging Face commit (`pqm_lora/models.py`):

- **SmilingWolf/wd-eva02-large-tagger-v3** (`b25b82a`), a booru tagger through
  onnxruntime on the GPU, since Illustrious speaks tags;
- **JoyCaption Beta One** (`fancyfeast/llama-joycaption-beta-one-hf-llava`,
  `ebf414e`), one sentence per image, bf16, greedy.

PQM's L.0 spike chose and measured both on an RTX 3090 Ti: the tagger 8.6 s for 40
images, JoyCaption 1.8 s an image with a 16.05 GB peak, so a 24 GB card.

`python -m pqm_lora.serve --port 7878` keeps both loaded (each on first use) and
answers on `127.0.0.1` only:

    GET  /health
    POST /tag      {"path": "/abs/image.png", "min_score": 0.2}
    POST /caption  {"path": "/abs/image.png", "prompt": "...", "max_new_tokens": 120}

The extension's `tag` and `caption` jobs ask it about one image at a time.

## Setting up a pod by hand

    bash setup/pod_setup.sh gpu
    bash setup/pod_setup.sh venv      # Python 3.11, torch 2.14.1 cu130, requirements-pod.txt
    bash setup/pod_setup.sh weights   # ~18 GB into $HF_HOME (/workspace/hf)
    bash setup/pod_setup.sh verify
    bash setup/pod_setup.sh serve

## Tests

The tests need no GPU and nothing from `requirements-pod.txt`: tag selection and
the service's routes run against fakes.

    python -m pytest
