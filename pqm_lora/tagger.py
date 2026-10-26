"""The booru tagger: SmilingWolf's wd-eva02-large-tagger-v3, through onnxruntime on the GPU.

Prepared as L.0's spike prepared it, which is the model's own recipe: the image
flattened onto white, padded to a square, resized to the model's input with
bicubic, and handed over as BGR floats 0-255, NHWC.

torch is imported before the session is made: onnxruntime's CUDA provider finds
the CUDA libraries torch's wheels bring once torch has loaded them (L.0 did the
same, and its tagger ran on CUDA).
"""

import time

from . import models, tags


class Tagger:
    def __init__(self):
        import torch  # noqa: F401 - loads the CUDA libraries for onnxruntime
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        def fetch(name):
            return hf_hub_download(models.TAGGER_REPO, name, revision=models.TAGGER_REVISION,
                                   local_files_only=True)

        self.session = ort.InferenceSession(
            fetch("model.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.provider = self.session.get_providers()[0]
        self.labels = tags.read_labels(fetch("selected_tags.csv"))
        self.size = int(self.session.get_inputs()[0].shape[1])
        self.input = self.session.get_inputs()[0].name
        self.model = models.label(models.TAGGER_REPO, models.TAGGER_REVISION)

    def tag(self, path, min_score=models.REPORT_MIN_SCORE):
        import numpy as np
        from PIL import Image

        started = time.time()
        with Image.open(path) as image:
            image = image.convert("RGBA")
            flat = Image.new("RGBA", image.size, (255, 255, 255, 255))
            flat.alpha_composite(image)
            rgb = flat.convert("RGB")
        side = max(rgb.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
        pixels = np.asarray(square.resize((self.size, self.size), Image.BICUBIC),
                            dtype=np.float32)[:, :, ::-1]
        scores = self.session.run(None, {self.input: pixels[None].copy()})[0][0]
        out = tags.select([float(s) for s in scores], self.labels, float(min_score))
        out.update({"model": self.model, "provider": self.provider,
                    "seconds": round(time.time() - started, 3)})
        return out
