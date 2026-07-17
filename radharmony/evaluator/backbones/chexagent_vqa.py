"""Generative CheXagent-2 backbone — visual question answering (VQA).

Wraps ``StanfordAIMI/CheXagent-2-3b`` as a question-conditioned answerer for
:class:`VQAEvaluator`. Unlike ``make_chexagent_generator`` (report
generation, which bakes in a fixed multi-step report prompt), this passes
each sample's **question** as the prompt.

Loading follows the official CheXagent-2 model card (see
``chexagent_gen`` for the verbatim recipe). Greedy decoding.

DEPENDENCY ISOLATION: CheXagent-2 pins ``transformers==4.40.0``; install the
``chexagent_gen`` extra in its own env.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import monai.transforms as mt
import torch

from ._utils import normalize_to_uint8, squeeze_frame_dim, to_3channel, to_pil_rgb

_CHEXAGENT_HUB = "StanfordAIMI/CheXagent-2-3b"
# Concise-answer instruction so closed (yes/no) questions get short answers.
_VQA_PROMPT_PREFIX = (
    "Answer the following question about the image with a very short, "
    "definitive, and concise answer (a single word when possible): "
)


def make_chexagent_vqa(
    device: str = "cuda",
    hub: str = _CHEXAGENT_HUB,
    prompt_prefix: str = _VQA_PROMPT_PREFIX,
    max_new_tokens: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    tmp_dir: str | None = None,
) -> tuple[Callable[[dict], dict], Callable[[list[str], list[str]], list[str]]]:
    """Return ``(transform, vqa_answerer)`` for generative CheXagent-2 VQA.

    transform : callable ``sample_dict -> sample_dict``
        Converts ``sample["img"]`` to a temp PNG and stores the PNG **path
        string** back under ``img`` (same contract as ``chexagent_gen``).
    vqa_answerer : callable ``(img_paths, questions) -> list[str]``
        Pairs with :class:`VQAEvaluator`.
    """
    from transformers import (  # lazy: needs transformers==4.40.0 (card)
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(hub, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hub, device_map="auto", trust_remote_code=True
    )
    model = model.to(dtype).eval()

    _tmp = tmp_dir or tempfile.mkdtemp(prefix="chexagent_vqa_png_")
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

    def _answer(path: str, question: str) -> str:
        query = tokenizer.from_list_format(
            [{"image": str(path)}, {"text": prompt_prefix + question}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        gen = model.generate(
            input_ids.to(model.device),
            do_sample=False, num_beams=1,
            temperature=1.0, top_p=1.0,
            use_cache=True, max_new_tokens=max_new_tokens,
        )[0]
        text = tokenizer.decode(gen[input_ids.size(1):-1])
        return text.replace("</s>", "").strip()

    @torch.no_grad()
    def vqa_answerer(img_paths: list[str], questions: list[str]) -> list[str]:
        return [_answer(str(p), str(q)) for p, q in zip(img_paths, questions)]

    return transform, vqa_answerer
