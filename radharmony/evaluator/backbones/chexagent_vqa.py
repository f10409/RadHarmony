"""Generative CheXagent-2 backbone for visual question answering (VQA).

Wraps ``StanfordAIMI/CheXagent-2-3b`` as a question-conditioned answerer for
:class:`VQAEvaluator`. Unlike ``make_chexagent_generator`` (report
generation, which uses a fixed multi-step report prompt), this passes
each sample's **question** as the prompt.

Loading follows the official CheXagent-2 model card (see
``chexagent_gen`` for the verbatim recipe). Greedy decoding.

DEPENDENCY ISOLATION: CheXagent-2 pins ``transformers==4.40.0``. Install the
``chexagent_gen`` extra in its own env.
"""

from __future__ import annotations

import os
from typing import Callable

import monai.transforms as mt
import torch

from ._utils import (
    display_255, load_image, png_cache_dir, png_path, squeeze_frame_dim, to_3channel, to_pil_rgb, to_uint8,
)

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
    # device_map=device keeps the whole model on device. "auto" can spread it over all
    # visible GPUs (V-B-15).
    model = AutoModelForCausalLM.from_pretrained(
        hub, device_map=device, trust_remote_code=True
    )
    model = model.to(dtype).eval()

    _tmp = png_cache_dir(tmp_dir, "chexagent_vqa_png_")

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
        return [_answer(str(p), str(q)) for p, q in zip(img_paths, questions, strict=True)]

    return transform, vqa_answerer
