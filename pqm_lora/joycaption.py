"""JoyCaption Beta One: one sentence about an image, in bf16 on the GPU.

As L.0's spike ran it: the model's chat template with a system line and the
prompt, greedy decoding (the same image gives the same sentence), at most 120
new tokens, the reply's whitespace folded. L.0 measured 16.05 GB of VRAM at the
peak and 1.8 s an image on an RTX 3090 Ti.
"""

import time

from . import models

SYSTEM = "You are a helpful image captioner."


class JoyCaption:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.torch = torch
        kwargs = {"revision": models.JOYCAPTION_REVISION, "local_files_only": True}
        self.processor = AutoProcessor.from_pretrained(models.JOYCAPTION_REPO, **kwargs)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            models.JOYCAPTION_REPO, torch_dtype=torch.bfloat16, device_map=0, **kwargs).eval()
        self.label = models.label(models.JOYCAPTION_REPO, models.JOYCAPTION_REVISION)

    def caption(self, path, prompt, max_new_tokens=models.MAX_NEW_TOKENS):
        from PIL import Image

        torch = self.torch
        started = time.time()
        convo = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(convo, tokenize=False,
                                                  add_generation_prompt=True)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        inputs = self.processor(text=[text], images=[rgb], return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                                      do_sample=False)[0]
        reply = self.processor.tokenizer.decode(out[inputs["input_ids"].shape[1]:],
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        return {"sentence": " ".join(reply.split()), "model": self.label,
                "seconds": round(time.time() - started, 3),
                "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
