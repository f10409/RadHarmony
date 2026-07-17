"""Generative MAIRA-2 backbone — report generation (Stage A of two-stage).

Microsoft Health Futures' chest-X-ray report generation model: a RAD-DINO
vision tower + Vicuna text decoder, purpose-built for free-text FINDINGS +
IMPRESSION generation. Distinct from MedImageInsights / MedSigLIP / RAD-DINO
which are contrastive *encoders* (no autoregressive decoder).

Loading follows the official `microsoft/maira-2` model card pattern (HF
custom-code model — `trust_remote_code=True` required). The single-image
"reporting input" path is used (no prior frontal/lateral, no indication
or technique — RadHarmony datasets currently surface only the image and
the reference report).

DEPENDENCY ISOLATION: MAIRA-2 needs ``transformers>=4.46`` while
``chexagent_gen`` pins ``transformers==4.40.0``. Run this in its own env
(the ``maira2_gen`` extra) — same two-stage workflow: generate pairs
parquet here, score with RadEval in the separate scoring env.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import monai.transforms as mt
import torch

from ._utils import normalize_to_uint8, squeeze_frame_dim, to_3channel, to_pil_rgb

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
        Loads ``sample["img"]`` (an absolute image path placed there by the
        RadHarmony dataset before transforms run) to a temp PNG and stores
        the PNG **path string** back under ``img``; passes ``report``
        through unchanged. Identical contract to ``chexagent_gen`` so the
        DICOM/MONAI loader works the same way.
    report_generator : callable ``list[str] -> list[str]``
        Input: PNG path strings. Output: generated free-text reports.
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

    _tmp = tmp_dir or tempfile.mkdtemp(prefix="maira2_png_")
    os.makedirs(_tmp, exist_ok=True)

    _loader = mt.Compose(
        [
            mt.LoadImage(image_only=True, ensure_channel_first=True),
            mt.Lambda(squeeze_frame_dim),
            mt.Transpose(indices=[0, 2, 1]),
            mt.Lambda(to_3channel),
            mt.Lambda(normalize_to_uint8),
        ]
    )

    def transform(sample: dict) -> dict:
        src = str(sample["img"])
        png = os.path.join(
            _tmp, os.path.splitext(os.path.basename(src))[0] + ".png"
        )
        if not os.path.exists(png):
            to_pil_rgb(_loader(src).float()).save(png)
        sample["img"] = png
        return sample

    @torch.no_grad()
    def report_generator(img_paths, indications=None) -> list[str]:
        """``indications`` (optional): per-image clinical-indication strings
        (``None`` entries → none), wired into MAIRA-2's native ``indication``
        reporting-input slot."""
        inds = indications if indications is not None else [None] * len(img_paths)
        out: list[str] = []
        for path, ind in zip(img_paths, inds):
            # --- official MAIRA-2 reporting recipe; isolated for easy correction ---
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
