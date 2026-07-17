"""Generative CheXagent-2 backbone — report generation (Stage A of two-stage).

Distinct from :func:`backbones.chexagent.make_chexagent` (XraySigLIP, a
contrastive *encoder* — no text generation).

Loading follows the **official StanfordAIMI/CheXagent-2-3b model card**
verbatim (verified 2026-05-19):

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
                name, device_map="auto", trust_remote_code=True).to(bf16).eval()
    query = tokenizer.from_list_format([{'image': path}, {'text': prompt}])
    conv  = [{"from":"system","value":"You are a helpful assistant."},
             {"from":"human","value":query}]
    ids   = tokenizer.apply_chat_template(conv, add_generation_prompt=True,
                                          return_tensors="pt")
    out   = model.generate(ids.to(device), do_sample=False, num_beams=1,
                           max_new_tokens=512)[0]
    resp  = tokenizer.decode(out[ids.size(1):-1])

CheXagent-2 consumes image **file paths**, not pixel tensors. So this
recipe's "transform" is a thin callable that converts each sample's image
(any MONAI-readable format incl. DICOM) to a temporary PNG and yields its
**path** under ``img`` (kept as a string — no ToTensor), leaving ``report``
untouched. The returned ``report_generator`` then takes a list of those
PNG paths.

DEPENDENCY ISOLATION: CheXagent-2 pins ``transformers==4.40.0`` (model card).
That conflicts with RadEval / RadHarmony's other extras, so this runs in its
**own env** (the ``chexagent_gen`` extra) and only produces (id, ref, hyp)
pairs — scoring (RadEval) happens in a separate env. See the two-stage
runbook in ``migration/HANDOFF.md``.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import monai.transforms as mt
import torch

from ._utils import normalize_to_uint8, squeeze_frame_dim, to_3channel, to_pil_rgb

_CHEXAGENT_GEN_HUB = "StanfordAIMI/CheXagent-2-3b"
_DEFAULT_PROMPT = (
    "Write the FINDINGS and IMPRESSION sections of a chest radiograph "
    "report for this image as a clinical narrative paragraph."
)  # A ready-made single-shot prompt: pass prompt=_DEFAULT_PROMPT (or your own
   # string) to make_chexagent_generator to use the single-call path. The
   # default (prompt=None) uses the multi-step ABCDE recipe below.

# Multi-step "ABCDE" recipe from the official Stanford-AIMI/CheXagent demo
# (commit e4f31e6e, demos/app_demo.py): one generation call per anatomy →
# concatenate findings → one text-only call for the impression. Produces
# free-text-style reports instead of the single-shot grounded markup.
_ANATOMIES = [
    "Airway",
    "Breathing",
    "Cardiac",
    "Diaphragm",
    "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)",
]
_FINDINGS_PROMPTS = [
    f'Please provide a detailed description of "{a}" in the chest X-ray'
    for a in _ANATOMIES
]
_IMPRESSION_PROMPT_TMPL = "Write the Impression section for the following Findings: {findings}"


def make_chexagent_generator(
    device: str = "cuda",
    hub: str = _CHEXAGENT_GEN_HUB,
    prompt: str | None = None,
    findings_prompts: list[str] | None = None,
    impression_prompt: str | None = None,
    max_new_tokens: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    tmp_dir: str | None = None,
) -> tuple[Callable[[dict], dict], Callable[..., list[str]]]:
    """Return ``(transform, report_generator)`` for generative CheXagent-2.

    Prompting
    ---------
    prompt :
        Controls the generation strategy:

        - ``None`` (default) → the multi-step **ABCDE** recipe (one call per
          anatomy → concatenate findings → one impression call), matching the
          official Stanford-AIMI demo.
        - a **string** → single-shot: one image call with that prompt (e.g.
          ``prompt=_DEFAULT_PROMPT`` or your own instruction).
    findings_prompts :
        Override the ABCDE per-anatomy findings prompts (list of strings).
        Only used on the multi-step path (``prompt is None``). Defaults to the
        demo's five anatomy prompts.
    impression_prompt :
        Override the impression template on the multi-step path. Must contain
        a ``{findings}`` placeholder. Defaults to the demo's template.

    In every case, if the evaluator passes a per-image indication (via
    ``use_indication=True``), it is prepended to the prompt(s) as clinical
    context.

    transform : callable ``sample_dict -> sample_dict``
        Converts ``sample["img"]`` (an absolute image path placed there by
        the RadHarmony dataset before transforms run) to a temp PNG and
        stores the PNG **path string** back under ``img``; passes
        ``report`` through unchanged. Plain callable (not a MONAI Compose)
        on purpose — we must keep ``img`` a string for CheXagent-2.
    report_generator : callable ``(img_paths, indications=None) -> list[str]``
        Input: PNG path strings (one per image) + optional indications.
        Output: generated reports.
    """
    findings_prompts = findings_prompts or _FINDINGS_PROMPTS
    impression_tmpl = impression_prompt or _IMPRESSION_PROMPT_TMPL
    from transformers import (  # lazy: needs transformers==4.40.0 (card)
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(hub, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hub, device_map="auto", trust_remote_code=True
    )
    model = model.to(dtype).eval()

    _tmp = tmp_dir or tempfile.mkdtemp(prefix="chexagent_png_")
    os.makedirs(_tmp, exist_ok=True)

    # Any MONAI-readable image (PNG/JPEG/DICOM/NIfTI) → uint8 3-channel.
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
        sample["img"] = png  # keep as string path — do NOT tensorize
        return sample

    def _one_gen(input_ids):
        gen = model.generate(
            input_ids.to(model.device),
            do_sample=False, num_beams=1,
            temperature=1.0, top_p=1.0,
            use_cache=True, max_new_tokens=max_new_tokens,
        )[0]
        text = tokenizer.decode(gen[input_ids.size(1):-1])
        # The demo's clean_text — drop trailing </s>.
        return text.replace("</s>", "").strip()

    def _ask_with_image(path: str, prompt_text: str) -> str:
        query = tokenizer.from_list_format(
            [{"image": str(path)}, {"text": prompt_text}]
        )
        conv = [{"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query}]
        return _one_gen(
            tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        )

    def _ask_text(prompt_text: str) -> str:
        query = tokenizer.from_list_format([{"text": prompt_text}])
        conv = [{"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query}]
        return _one_gen(
            tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        )

    @torch.no_grad()
    def report_generator(img_paths, indications=None) -> list[str]:
        """Official Stanford-AIMI/CheXagent demo recipe (`demos/app_demo.py`):
        step 1 — 5 anatomy-specific findings calls + concatenate; step 2 —
        impression call (text-only) over the concatenated findings.
        Produces free-text-style reports, NOT the single-shot grounded
        markup the original prompt yielded.

        ``indications`` (optional): a per-image list of clinical-indication
        strings (``None`` entries → none). When present, the indication is
        prepended to each anatomy findings prompt as clinical context.
        """
        inds = indications if indications is not None else [None] * len(img_paths)
        out: list[str] = []
        for path, ind in zip(img_paths, inds):
            prefix = f"Clinical indication: {ind}. " if ind else ""

            if prompt is not None:
                # Single-shot path: one image call with the custom prompt.
                out.append(_ask_with_image(str(path), prefix + prompt))
                continue

            # Step 1: ABCDE findings (skip the "Determine the view" prefix —
            # the demo discards its output, but we still keep parity by not
            # accumulating it).
            findings_parts = []
            for fp in findings_prompts:
                findings_parts.append(_ask_with_image(str(path), prefix + fp))
            findings = " ".join(p for p in findings_parts if p).strip()

            # Step 2: impression (text-only call over the concatenated findings)
            impression = _ask_text(impression_tmpl.format(findings=findings))

            out.append(f"FINDINGS: {findings}\n\nIMPRESSION: {impression}")
        return out

    return transform, report_generator
