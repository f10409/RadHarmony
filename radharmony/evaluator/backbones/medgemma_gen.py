"""Generative MedGemma-4B backbone for report generation (Stage A of two-stage).

Wraps ``google/medgemma-4b-it`` (Gemma-3-based medical VLM) as a free-text
chest-X-ray report generator for :class:`ReportGenerationEvaluator`. Companion
to :func:`make_medgemma_vqa` (same model, question-conditioned). This recipe
produces FINDINGS-style narrative instead of a short VQA answer.

Prompting follows the MedGemma technical report's CXR reporting setup. The
model gets the exam **indication** and is asked for the findings, that is,
the ``"<INDICATION> findings:"`` form. This recipe makes it a ``prompt``
parameter that you can change (default :data:`_REPORT_PROMPT`). When the
evaluator passes a per-image indication (``use_indication=True``), the recipe
prepends it as ``"Indication: <ind>. "`` before the prompt.

Decoding is greedy (temperature 0), single run. This matches the report's
deterministic protocol.

``google/medgemma-4b-it`` is a **gated** Hugging Face model. Accept the
license on the model page and authenticate (``huggingface-cli login``)
before first use.

DEPENDENCY ISOLATION: MedGemma needs a recent ``transformers`` (Gemma-3
support). Install the ``medgemma_gen`` extra in its own env. It conflicts
with ``chexagent_gen`` (transformers==4.40.0) and ``maira2_gen``
(transformers<4.47). Same two-stage workflow as the other generative
recipes: produce the (id, ref, hyp) pairs parquet here, score with RadEval
in the separate scoring env.
"""

from __future__ import annotations

import os
from typing import Callable

import monai.transforms as mt
import torch

from ._utils import (
    display_255, load_image, png_cache_dir, png_path, squeeze_frame_dim, to_3channel, to_pil_rgb, to_uint8,
)

_MEDGEMMA_GEN_HUB = "google/medgemma-4b-it"
# CXR report-generation prompt, MedGemma technical report ("<INDICATION>
# findings:" form). Asks for the FINDINGS narrative. When an indication is
# given, report_generator prepends it as clinical context.
_REPORT_PROMPT = (
    "Given this chest X-ray, write the findings section of the radiology "
    "report as a clinical narrative."
)


def make_medgemma_generator(
    device: str = "cuda",
    hub: str = _MEDGEMMA_GEN_HUB,
    prompt: str = _REPORT_PROMPT,
    system_message: str | None = None,
    max_new_tokens: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    tmp_dir: str | None = None,
) -> tuple[Callable[[dict], dict], Callable[..., list[str]]]:
    """Return ``(transform, report_generator)`` for generative MedGemma-4B.

    Prompting
    ---------
    prompt :
        The instruction sent with each image. Defaults to
        :data:`_REPORT_PROMPT` (the report's ``"<INDICATION> findings:"``
        form). Pass your own string to override it (for example, to ask for an
        impression, or a combined findings+impression report).
    system_message :
        Optional persona system message. ``None`` (default) matches the
        MedGemma-faithful setting (no persona prefix, see
        :func:`make_medgemma_vqa`).

    In every case, if the evaluator passes a per-image indication (via
    ``use_indication=True``), it is prepended to the prompt as
    ``"Indication: <ind>. "`` clinical context.

    transform : callable ``sample_dict -> sample_dict``
        Converts ``sample["img"]`` to a temp PNG. ``sample["img"]`` is an
        absolute image path that the RadHarmony dataset puts there before
        transforms run. Stores the PNG **path string** back under ``img``.
        Passes ``report`` through unchanged. Same contract as the other generative
        recipes so the DICOM/MONAI loader works the same.
    report_generator : callable ``(img_paths, indications=None) -> list[str]``
        Input: PNG path strings (one per image) + optional indications.
        Output: generated report text.
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

    _tmp = png_cache_dir(tmp_dir, "medgemma_gen_png_")

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

    def _generate_one(path: str, prompt_text: str) -> str:
        # Official MedGemma model-card chat pattern, greedy decoding.
        content = [
            {"type": "image", "image": Image.open(str(path)).convert("RGB")},
            {"type": "text", "text": prompt_text},
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
    def report_generator(img_paths: list[str], indications: list[str | None] | None = None) -> list[str]:
        """Return one report per image. ``indications`` is optional: one indication text per
        image (``None`` for none). It goes before the prompt as ``"Indication: <ind>. "``."""
        inds = indications if indications is not None else [None] * len(img_paths)
        out: list[str] = []
        for path, ind in zip(img_paths, inds, strict=True):
            prefix = f"Indication: {ind}. " if ind else ""
            out.append(_generate_one(str(path), prefix + prompt))
        return out

    return transform, report_generator
