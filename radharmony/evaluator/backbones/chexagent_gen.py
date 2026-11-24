"""Generative CheXagent-2 backbone, report generation (Stage A of two-stage).

Distinct from :func:`backbones.chexagent.make_chexagent` (XraySigLIP, a
contrastive *encoder*, no text generation).

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

One change: the recipe loads with ``device_map=device``, not ``"auto"``. So
the model stays on the ``device`` argument (V-B-15).

CheXagent-2 consumes image **file paths**, not pixel tensors. So this
recipe's "transform" is a thin callable that converts each sample's image
(any MONAI-readable format incl. DICOM) to a temporary PNG and yields its
**path** under ``img`` (kept as a string, no ToTensor), leaving ``report``
untouched. The returned ``report_generator`` then takes a list of those
PNG paths.

DEPENDENCY ISOLATION: CheXagent-2 pins ``transformers==4.40.0`` (model card).
That conflicts with RadEval / RadHarmony's other extras. So this runs in its
**own env** (the ``chexagent_gen`` extra) and only makes (id, ref, hyp)
pairs. Scoring (RadEval) happens in a separate env. See the two-stage
runbook in ``migration/HANDOFF.md``.
"""

from __future__ import annotations

import os
import warnings
from typing import Callable

import monai.transforms as mt
import torch

from ._utils import (
    display_255, load_image, png_cache_dir, png_path, squeeze_frame_dim, to_3channel, to_pil_rgb, to_uint8,
)

_CHEXAGENT_GEN_HUB = "StanfordAIMI/CheXagent-2-3b"

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
        ``None`` (default) runs the multi-step ABCDE recipe of the official
        Stanford-AIMI demo: one call per anatomy, then one impression call over
        the joined findings. A string makes one call per image with that prompt
        and returns the answer as it is.
    findings_prompts :
        Replace the 5 ABCDE findings prompts (list of strings). Used only when
        ``prompt is None``. Default: the 5 anatomy prompts of the demo.
    impression_prompt :
        Replace the impression template. Used only when ``prompt is None``. It
        must hold a ``{findings}`` placeholder. Default: the demo template.

    With ``use_indication=True``, the evaluator gives an indication for each image.
    The generator puts it before each image prompt as clinical context: the 5
    findings prompts, or ``prompt``. The impression call does not get it.

    transform : callable ``sample_dict -> sample_dict``
        Converts ``sample["img"]`` (an absolute image path) to a temp PNG
        and stores the PNG **path string** back under ``img``. Passes
        ``report`` through unchanged. Plain callable (not a MONAI Compose)
        on purpose. We must keep ``img`` a string for CheXagent-2.
    report_generator : callable ``(img_paths, indications=None) -> list[str]``
        Input: PNG path strings (one per image), and an optional indication
        for each image. Output: generated reports.
    """
    if prompt is not None and (findings_prompts is not None or impression_prompt is not None):
        warnings.warn("findings_prompts and impression_prompt are not used when prompt is set (D-52).",
                      UserWarning, stacklevel=2)
    findings_prompts = _FINDINGS_PROMPTS if findings_prompts is None else list(findings_prompts)
    if not findings_prompts:
        raise ValueError("findings_prompts is empty. Pass None for the 5 prompts of the demo.")
    impression_tmpl = _IMPRESSION_PROMPT_TMPL if impression_prompt is None else impression_prompt
    if "{findings}" not in impression_tmpl:
        raise ValueError("impression_prompt must hold a {findings} placeholder.")
    from transformers import (  # lazy: needs transformers==4.40.0 (card)
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(hub, trust_remote_code=True)
    # The model card uses device_map="auto", which can spread the model over all
    # visible GPUs. device_map=device keeps the whole model on device (V-B-15).
    model = AutoModelForCausalLM.from_pretrained(
        hub, device_map=device, trust_remote_code=True
    )
    model = model.to(dtype).eval()

    _tmp = png_cache_dir(tmp_dir, "chexagent_png_")

    # Any MONAI-readable image (PNG/JPEG/DICOM/NIfTI) → uint8 3-channel.
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

    def _one_gen(input_ids):
        gen = model.generate(
            input_ids.to(model.device),
            do_sample=False, num_beams=1,
            temperature=1.0, top_p=1.0,
            use_cache=True, max_new_tokens=max_new_tokens,
        )[0]
        text = tokenizer.decode(gen[input_ids.size(1):-1])
        # The demo's clean_text, drop trailing </s>.
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
    def report_generator(img_paths: list[str], indications: list[str | None] | None = None) -> list[str]:
        """Official Stanford-AIMI/CheXagent demo recipe (`demos/app_demo.py`):
        Step 1: 5 anatomy-specific findings calls + concatenate. Step 2:
        impression call (text-only) over the concatenated findings.
        Produces free-text-style reports, NOT the single-shot grounded
        markup the original prompt yielded.

        ``indications`` is optional: one indication text per image (``None``
        for none). It goes before each image prompt. The impression call does not get it.
        """
        inds = indications if indications is not None else [None] * len(img_paths)
        out: list[str] = []
        for path, ind in zip(img_paths, inds, strict=True):
            prefix = f"Clinical indication: {ind}. " if ind else ""

            if prompt is not None:
                # One call per image with the given prompt. The answer is kept as it is.
                out.append(_ask_with_image(str(path), prefix + prompt))
                continue

            # Step 1: ABCDE findings (skip the "Determine the view" prefix):
            # the demo discards its output, but we still keep parity by not
            # accumulating it.
            findings_parts = []
            for fp in findings_prompts:
                findings_parts.append(_ask_with_image(str(path), prefix + fp))
            findings = " ".join(p for p in findings_parts if p).strip()

            # Step 2: impression (text-only call over the concatenated findings)
            impression = _ask_text(impression_tmpl.format(findings=findings))

            out.append(f"FINDINGS: {findings}\n\nIMPRESSION: {impression}")
        return out

    return transform, report_generator
