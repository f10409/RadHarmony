"""Generative MedGemma-4B backbone for visual question answering (VQA).

Wraps ``google/medgemma-4b-it`` (Gemma-3-based medical VLM) as a
question-conditioned answerer for :class:`VQAEvaluator`. This is the recipe
to use when **replicating the MedGemma technical report's VQA-RAD / SLAKE
numbers** (report §3.4, Table 9): the answerer runs the model greedily
(temperature 0) with the report's VQA-RAD prompt.

Faithfulness notes (from the report):
- Model: MedGemma 4B multimodal, instruction-tuned (``-it``).
- Decoding: greedy, single run (§3.1, temperature 0.0).
- Prompt: the VQA-RAD prompt in Appendix Table A7. The table's caption
  says that the "You are a helpful radiology assistant" system message was
  prefixed for *all but the medical MedGemma and Med-Gemini models*. So the
  MedGemma-faithful setting uses **no** persona system message
  (``system_message=None``, the default here).

``google/medgemma-4b-it`` is a **gated** Hugging Face model. Accept the
license on the model page and authenticate (``huggingface-cli login``)
before first use.

DEPENDENCY ISOLATION: MedGemma needs a recent ``transformers`` (Gemma-3
support). Install the ``medgemma_gen`` extra in its own env. It conflicts
with ``chexagent_gen`` (transformers==4.40.0) and ``maira2_gen``
(transformers<4.47).
"""

from __future__ import annotations

import os
from typing import Callable

import monai.transforms as mt
import torch

from ._utils import (
    display_255, load_image, png_cache_dir, png_path, squeeze_frame_dim, to_3channel, to_pil_rgb, to_uint8,
)

_MEDGEMMA_VQA_HUB = "google/medgemma-4b-it"
# VQA-RAD prompt, MedGemma technical report Appendix Table A7 (verbatim).
_VQA_RAD_PROMPT = (
    "Given this radiology image, which can be a frontal chest X-ray, a single "
    "slice head or abdominal CT or MR image, provide a very short, definitive, "
    "and concise answer (if possible, a single word) to the following question: "
)


def make_medgemma_vqa(
    device: str = "cuda",
    hub: str = _MEDGEMMA_VQA_HUB,
    prompt: str = _VQA_RAD_PROMPT,
    system_message: str | None = None,
    max_new_tokens: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    tmp_dir: str | None = None,
) -> tuple[Callable[[dict], dict], Callable[[list[str], list[str]], list[str]]]:
    """Return ``(transform, vqa_answerer)`` for generative MedGemma-4B VQA.

    transform : callable ``sample_dict -> sample_dict``
        Converts ``sample["img"]`` to a temp PNG. ``sample["img"]`` is an
        absolute image path that the RadHarmony dataset puts there before
        transforms run. Stores the PNG **path string** back under ``img``.
        Passes ``question`` / ``answer`` through unchanged. Same contract as the
        report-generation recipes so the DICOM/MONAI loader works the same.
    vqa_answerer : callable ``(img_paths, questions) -> list[str]``
        Input: PNG path strings + the per-image questions. Output: the
        generated answers. Pair with :class:`VQAEvaluator`.
    """
    from PIL import Image  # lazy
    from transformers import (  # lazy: needs a Gemma-3-capable transformers
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(hub)
    # device_map=device keeps the whole model on device. "auto" can spread it over all
    # visible GPUs (V-B-15).
    model = AutoModelForImageTextToText.from_pretrained(
        hub, torch_dtype=dtype, device_map=device
    )
    model = model.eval()

    _tmp = png_cache_dir(tmp_dir, "medgemma_png_")

    _loader = mt.Compose(
        [
            load_image(),
            mt.Lambda(squeeze_frame_dim),
            mt.Transpose(indices=[0, 2, 1]),
            mt.Lambda(display_255),
            mt.Lambda(to_3channel),
            mt.Lambda(to_uint8),
        ]
    )

    def transform(sample: dict) -> dict:
        src = str(sample["img"])
        png = png_path(_tmp, src)
        if not os.path.exists(png):
            to_pil_rgb(_loader(src).float()).save(png)
        sample["img"] = png  # keep as string path, do NOT tensorize
        return sample

    def _answer(path: str, question: str) -> str:
        # Official MedGemma model-card chat pattern, greedy decoding.
        content = [
            {"type": "image", "image": Image.open(str(path)).convert("RGB")},
            {"type": "text", "text": prompt + question},
        ]
        messages = []
        if system_message:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_message}]}
            )
        messages.append({"role": "user", "content": content})
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=dtype)
        prompt_len = inputs["input_ids"].shape[-1]
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        text = processor.decode(gen[0][prompt_len:], skip_special_tokens=True)
        return text.strip()

    @torch.no_grad()
    def vqa_answerer(img_paths: list[str], questions: list[str]) -> list[str]:
        return [_answer(str(p), str(q)) for p, q in zip(img_paths, questions, strict=True)]

    return transform, vqa_answerer
