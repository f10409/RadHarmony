"""Backbone recipe modules for the radharmony evaluator.

Each module exposes a ``make_*`` factory that returns a ready-to-use
transform and encoder (and, for vision-language models, a text encoder
+ tokenizer). Imports inside each factory are **lazy** so this package
can be imported even if the backbone's optional dependency isn't installed.

Available factories
-------------------
- :func:`~radharmony.evaluator.backbones.raddino.make_raddino`
  RAD-DINO on pre-processed PNGs. Requires the ``raddino`` extra::

      uv pip install -e ".[raddino]"

- :func:`~radharmony.evaluator.backbones.biomed_clip.make_biomed_clip`
  BiomedCLIP. Requires the ``biomed`` extra::

      uv pip install -e ".[biomed]"

- :func:`~radharmony.evaluator.backbones.chexagent.make_chexagent`
  XraySigLIP (StanfordAIMI), 1024-d image embeddings, 512×512 input.
  Returns ``(transform, image_encoder, text_encoder, processor)``.
  Requires the ``chexagent`` extra::

      uv pip install -e ".[chexagent]"

- :func:`~radharmony.evaluator.backbones.medsiglip.make_medsiglip`
  MedSigLIP (Google), 1152-d pooled CLS embeddings, 448×448 input.
  Returns ``(transform, image_encoder, text_encoder, processor)``.
  Requires the ``medsiglip`` extra::

      uv pip install -e ".[medsiglip]"

- :func:`~radharmony.evaluator.backbones.medimageinsights.make_medimageinsights`
  MedImageInsights (Microsoft / Lion-AI), 512-d L2-normalised embeddings,
  480×480 input (DaViT, 360M params). Returns
  ``(transform, image_encoder, text_encoder, None)``.
  Requires the ``medimageinsights`` extra::

      uv pip install -e ".[medimageinsights]"

- :func:`~radharmony.evaluator.backbones.chexfound.make_chexfound`
  CheXFound (ViT-L/16, iBOT-style), 1024-d CLS embeddings, 512×512 input.
  Returns ``(transform, image_encoder)``. Requires the ``chexfound`` extra
  and a manually downloaded checkpoint::

      uv pip install -e ".[chexfound]"

- :func:`~radharmony.evaluator.backbones.dinov3.make_dinov3`
  DINOv3 (Meta), ViT-B/16 by default — 768-d CLS embeddings, 224×224 input,
  4 register tokens skipped in segmentation mode. Returns
  ``(transform, encoder)``. Requires the ``dinov3`` extra::

      uv pip install -e ".[dinov3]"

- :func:`~radharmony.evaluator.backbones.eva_x.make_eva_x`
  EVA-X (HUST VL), CXR-pretrained ViT — tiny/small/base variants
  (192/384/768-d CLS embeddings), 224×224 input. Returns
  ``(transform, encoder)``. Requires the ``eva_x`` extra and the
  vendored ``third_party_models/EVA-X`` submodule::

      git submodule update --init third_party_models/EVA-X
      uv pip install -e ".[eva_x]"

- :func:`~radharmony.evaluator.backbones.ark_plus.make_ark_plus`
  Ark+ (jlianglab), CXR-pretrained Swin-L — 1536-d ``forward_features``
  embeddings, 768×768 input. Returns ``(transform, encoder)``. Requires
  manual checkpoint download and side-installs ``timm==0.5.4`` into
  ``third_party_models/Ark/timm-054`` on first call (the venv's timm is
  untouched)::

      uv pip install -e ".[ark_plus]"

- :func:`~radharmony.evaluator.backbones.medical_mae.make_medical_mae`
  Medical MAE (lambert-x), CXR-pretrained MAE ViT — 768-d (base) or 384-d
  (small) CLS embeddings, 224×224 input with chestx-ray normalization
  (``[0.5056]*3 / [0.252]*3``). Returns ``(transform, encoder)``. Requires
  the cloned ``lambert-x/medical_mae`` repo, a manually downloaded
  checkpoint, and side-installs ``timm==0.4.12`` into
  ``third_party_models/medical_mae/timm-0412`` on first call::

      uv pip install -e ".[medical_mae]"

- :func:`~radharmony.evaluator.backbones.siglip2.make_siglip2`
  SigLIP 2 (Google), general-domain vision-language SigLIP — 1152-d pooled
  embeddings (so400m), 384×384 input. Returns
  ``(transform, image_encoder, text_encoder, processor)``. Uses Option B
  (``m.vision_model(pixel_values=x).pooler_output``) and skips the spurious
  CLS slice (SigLIP has no CLS token). Requires the ``siglip2`` extra::

      uv pip install -e ".[siglip2]"
"""

from .raddino import make_raddino
from .biomed_clip import make_biomed_clip
from .chexagent import make_chexagent
from .medsiglip import make_medsiglip
from .medimageinsights import make_medimageinsights
from .chexfound import make_chexfound
from .dinov3 import make_dinov3
from .eva_x import make_eva_x
from .ark_plus import make_ark_plus
from .medical_mae import make_medical_mae
from .siglip2 import make_siglip2

__all__ = ["make_raddino", "make_biomed_clip", "make_chexagent", "make_medsiglip", "make_medimageinsights", "make_chexfound", "make_dinov3", "make_eva_x", "make_ark_plus", "make_medical_mae", "make_siglip2"]
