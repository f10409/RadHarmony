"""Generative MAIRA-2 backbone, report generation (Stage A of two-stage).

Microsoft Health Futures' chest-X-ray report generation model: a RAD-DINO
vision tower + Vicuna text decoder, purpose-built for free-text FINDINGS +
IMPRESSION generation. Distinct from MedImageInsights / MedSigLIP / RAD-DINO
which are contrastive *encoders* (no autoregressive decoder).

Loading follows the official `microsoft/maira-2` model card pattern (HF
custom-code model, `trust_remote_code=True` required). The single-image
"reporting input" path is used (no prior frontal or lateral image, no
technique). The indication is used only when the evaluator gives it
(``use_indication=True``).

DEPENDENCY ISOLATION: MAIRA-2 needs ``transformers>=4.46`` while
``chexagent_gen`` pins ``transformers==4.40.0``. Run this in its own env
(the ``maira2_gen`` extra), same two-stage workflow: generate pairs
parquet here, score with RadEval in the separate scoring env.
"""

from __future__ import annotations

import os
from typing import Callable

import monai.transforms as mt
import torch

from ._utils import (
    display_255, load_image, png_cache_dir, png_path, squeeze_frame_dim, to_3channel, to_pil_rgb, to_uint8,
)

_MAIRA2_HUB = "microsoft/maira-2"


def make_maira2_generator(
    device: str = "cuda",
    hub: str = _MAIRA2_HUB,
    max_new_tokens: int = 300,
    dtype: torch.dtype = torch.bfloat16,
    tmp_dir: str | None = None,
    get_grounding: bool = False,
) -> tuple[Callable[[dict], dict], Callable[..., list[str]]]:
    """Return ``(transform, report_generator)`` for generative MAIRA-2.

    transform : callable ``sample_dict -> sample_dict``
        Loads ``sample["img"]`` (an absolute image path) to a temp PNG
        and stores the PNG **path string** back under ``img``. Passes ``report``
        through unchanged. Identical contract to ``chexagent_gen`` so the
        DICOM/MONAI loader works the same way.
    report_generator : callable ``(img_paths, indications=None) -> list[str]``
        Input: PNG path strings, and an optional indication for each image.
        Output: generated free-text reports.
    """
    from PIL import Image  # lazy
    from transformers import (  # lazy: needs transformers>=4.46
        AutoModelForCausalLM,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(hub, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hub, trust_remote_code=True, torch_dtype=dtype
    )
    model = model.to(device).eval()

    _tmp = png_cache_dir(tmp_dir, "maira2_png_")

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
        sample["img"] = png
        return sample

    @torch.no_grad()
    def report_generator(img_paths: list[str], indications: list[str | None] | None = None) -> list[str]:
        """Return one report per image. ``indications`` is optional: one indication text per
        image (``None`` for none). It goes into the MAIRA-2 ``indication`` input."""
        inds = indications if indications is not None else [None] * len(img_paths)
        out: list[str] = []
        for path, ind in zip(img_paths, inds, strict=True):
            # --- official MAIRA-2 reporting recipe. Isolated for easy correction ---
            img = Image.open(str(path)).convert("RGB")
            inputs = processor.format_and_preprocess_reporting_input(
                current_frontal=img,
                current_lateral=None,
                prior_frontal=None,
                indication=(ind or None),
                technique=None,
                comparison=None,
                prior_report=None,
                return_tensors="pt",
                get_grounding=get_grounding,
            )
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[-1]
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                num_beams=1,
            )
            text = processor.decode(gen[0][prompt_len:], skip_special_tokens=True)
            out.append(text.strip())
            # ----------------------------------------------------------------
        return out

    return transform, report_generator
