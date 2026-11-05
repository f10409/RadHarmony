"""Backbone recipes give the input and the feature of the authors' code (D-16).

These tests need the model venv (torchvision, transformers). Tests that need
weights run only where the weights are in the Hugging Face cache. Run with::

    PYTHONPATH=$PWD /path/to/model/.venv/bin/python -m pytest tests/test_backbones.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch


def _gray(h: int, w: int, seed: int = 0) -> np.ndarray:
    return (np.random.RandomState(seed).rand(h, w) * 255).astype(np.uint8)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


# --------------------------------------------------------------------------
# V-B-2: Ark+ resizes to 896, then center-crops 768
# --------------------------------------------------------------------------
def test_ark_plus_image_geometry_matches_the_authors():
    """Ark_Plus/Finetuning/dataloader.py build_transform_classification(mode="valid"),
    with the README fine-tuning arguments --img_size 896 --input_size 768."""
    T = pytest.importorskip("torchvision.transforms")
    from PIL import Image

    from radharmony.evaluator.backbones.ark_plus import _image_transform

    gray = _gray(300, 400)
    official = T.Compose([
        T.Resize((896, 896)), T.CenterCrop(768), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(Image.fromarray(gray).convert("RGB"))
    x = torch.from_numpy(gray).float().div(255).expand(3, -1, -1)
    got = _image_transform()(x)
    assert got.shape == official.shape == (3, 768, 768)
    assert _cos(got, official) > 0.999
    assert (got - official).abs().mean() < 0.01


def test_ark_plus_mask_follows_the_image_crop():
    pytest.importorskip("torchvision")
    from radharmony.evaluator.backbones.ark_plus import _mask_transform

    mask = torch.zeros(1, 300, 400)
    mask[:, :, :50] = 1  # left 50 of 400 columns: outside the center crop
    out = _mask_transform()(mask)
    assert out.shape == (1, 768, 768)
    # Resize 400 -> 896 puts the edge at 112 px. The crop removes 64 px per side.
    assert out[0, 0, :40].eq(1).all() and out[0, 0, 60:].eq(0).all()


def test_ark_plus_classification_feature_is_the_projector_output():
    """Q-2 (D-16): the authors' linear probe uses generate_embeddings(x, after_proj=True).

    Its emb_dim is 1376 (Ark_Plus/Linearprobing/main_linearprobing.py). generate_embeddings
    is projector(forward_features(x)). The projector is one Linear of the checkpoint teacher."""
    import os

    pytest.importorskip("torchvision")
    from radharmony.evaluator.backbones.ark_plus import _DEFAULT_CHECKPOINT_PATH, make_ark_plus

    if not os.path.exists(_DEFAULT_CHECKPOINT_PATH):
        pytest.skip("Ark+ checkpoint not on this machine")
    teacher = torch.load(_DEFAULT_CHECKPOINT_PATH, map_location="cpu", weights_only=False)["teacher"]
    w, b = teacher["module.projector.weight"], teacher["module.projector.bias"]
    assert tuple(w.shape) == (1376, 1536)
    _, encoder = make_ark_plus(device="cpu", dtype=torch.float32)
    x = torch.randn(1, 3, 768, 768, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        got = encoder(x)
        expected = torch.nn.functional.linear(encoder.model.forward_features(x), w, b)
    assert got.shape == (1, 1376)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# V-B-10: to_pil_rgb keeps the color of a 3-channel input. V-B-12: its input is
# in 0 to 255, and it keeps the values (D-27).
# --------------------------------------------------------------------------
def test_to_pil_rgb_keeps_three_channels():
    from radharmony.evaluator.backbones._utils import to_pil_rgb

    x = torch.zeros(3, 4, 4)
    x[1] = 127.0
    x[2] = 255.0
    out = np.asarray(to_pil_rgb(x))
    assert out.shape == (4, 4, 3)
    assert out[..., 0].mean() == 0 and out[..., 1].mean() == 127 and out[..., 2].mean() == 255


def test_to_pil_rgb_gray_is_unchanged():
    from radharmony.evaluator.backbones._utils import to_pil_rgb

    expected = _gray(5, 6) // 2 + 10  # 10 to 137: a stretch would change it
    gray = torch.from_numpy(expected).float()[None]
    for x in (gray, gray.expand(3, -1, -1)):
        out = np.asarray(to_pil_rgb(x))
        for c in range(3):
            np.testing.assert_array_equal(out[..., c], expected)


# --------------------------------------------------------------------------
# MedImageInsights: eval mode (V-B-5) and the text device (V-B-9)
# --------------------------------------------------------------------------
def _mii_cached() -> bool:
    hub = pytest.importorskip("huggingface_hub")
    return isinstance(hub.try_to_load_from_cache("lion-ai/MedImageInsights",
                                                 "medimageinsightmodel.py"), str)


@pytest.fixture(scope="module")
def mii_cpu():
    pytest.importorskip("mup")
    pytest.importorskip("torchvision")
    if not _mii_cached():
        pytest.skip("MedImageInsights snapshot not cached")
    from radharmony.evaluator.backbones import make_medimageinsights
    return make_medimageinsights(device="cpu", dtype=torch.float32)


def test_medimageinsights_encoder_is_in_eval_mode(mii_cpu):
    _, encoder, _, _ = mii_cpu
    assert not any(m.training for m in encoder.model.modules())
    x = torch.randn(1, 3, 480, 480, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.testing.assert_close(encoder(x), encoder(x))


def test_medimageinsights_text_runs_on_the_model_device(mii_cpu):
    _, _, text_encoder, _ = mii_cpu
    emb = text_encoder(["pneumonia", "no finding"])
    assert emb.shape == (2, 1024)
    torch.testing.assert_close(emb, text_encoder(["pneumonia", "no finding"]))


# --------------------------------------------------------------------------
# V-B-13: the CheXFound feature joins the CLS tokens of the last 4 blocks.
# The authors' frozen-feature eval does this (linear_glori.py: n_last_blocks=[4],
# avgpools=[False]). Their create_linear_input joins the tokens in block order.
# --------------------------------------------------------------------------
class _FakeViT(torch.nn.Module):
    """Act as the CheXFound ViT: block i gives the CLS token i (8 values)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def load_state_dict(self, state_dict, strict=True):
        return None

    def get_intermediate_layers(self, x, n=1, return_class_token=False, norm=True):
        layers = [(torch.zeros(len(x), 4, 8), torch.full((len(x), 8), float(i))) for i in range(24)]
        return tuple(layers[-n:])

    def forward(self, x):
        return torch.full((len(x), 8), 23.0)  # the CLS token of the last block


_LAST_4_CLS = torch.cat([torch.full((2, 8), float(i)) for i in (20, 21, 22, 23)], dim=-1)


def test_chexfound_feature_is_the_cls_of_the_last_4_blocks():
    from radharmony.evaluator.backbones.chexfound import _cls_of_last_4_blocks

    torch.testing.assert_close(_cls_of_last_4_blocks(_FakeViT(), torch.zeros(2, 3, 16, 16)), _LAST_4_CLS)


def test_chexfound_encoder_gives_the_cls_of_the_last_4_blocks(tmp_path, monkeypatch):
    import sys
    import types

    pytest.importorskip("torchvision")
    vt = types.ModuleType("chexfound.models.vision_transformer")
    vt.vit_large = lambda **kwargs: _FakeViT()
    for name in ("chexfound", "chexfound.models"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "chexfound.models.vision_transformer", vt)
    (tmp_path / "weights").mkdir()
    torch.save({"teacher": {}}, tmp_path / "weights" / "teacher_checkpoint.pth")

    from radharmony.evaluator.backbones import make_chexfound

    _, encoder = make_chexfound(chexfound_dir=str(tmp_path), device="cpu", dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(encoder(torch.zeros(2, 3, 16, 16)), _LAST_4_CLS)


def test_chexfound_input_is_rescaled_to_0_1_as_the_authors_do():
    """V-B-14: CheXFound/chexfound/data/transforms.py make_classification_eval_transform runs
    Resize(512, BICUBIC), CenterCrop(512), ToTensor, RescaleImage (min-max of each channel), Normalize.
    So each channel is in [0, 1] before Normalize, also when the bicubic resize overshoots."""
    T = pytest.importorskip("torchvision.transforms")
    from PIL import Image

    from radharmony.evaluator.backbones.chexfound import _image_transform

    gray = np.zeros((300, 400), np.uint8)
    gray[:, 100:250] = 255  # sharp edges: a bicubic resize overshoots below 0 and above 1
    gray[50:80, 300:380] = 128
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    official = T.Compose([
        T.Resize(512, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(512), T.ToTensor(),
    ])(Image.fromarray(gray).convert("RGB"))
    lo = official.amin(dim=(1, 2), keepdim=True)
    official = (official - lo) / (official.amax(dim=(1, 2), keepdim=True) - lo)  # RescaleImage
    x = torch.from_numpy(gray).float().div(255).expand(3, -1, -1)
    got = _image_transform()(x) * std + mean  # undo Normalize
    torch.testing.assert_close(got.amin(dim=(1, 2)), torch.zeros(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(got.amax(dim=(1, 2)), torch.ones(3), atol=1e-5, rtol=0)
    assert (got - official).abs().mean() < 0.01


def test_eva_x_missing_repo_names_the_clone_command(tmp_path):
    """A-14: the repo has no git submodule for EVA-X, so the message gives the clone command."""
    pytest.importorskip("torchvision")
    from radharmony.evaluator.backbones import make_eva_x

    with pytest.raises(FileNotFoundError, match="git clone https://github.com/hustvl/EVA-X"):
        make_eva_x(repo_path=str(tmp_path / "EVA-X"), device="cpu")


# --------------------------------------------------------------------------
# V-B-8: SigLIP2 text is lowercased, as in the SigLIP2 training
# --------------------------------------------------------------------------
def test_siglip2_processor_lowercases_text():
    """transformers' own Siglip2Tokenizer adds a Lowercase normalizer:
    "Gemma tokenizer + SigLIP2 training default: lowercase normalization"."""
    transformers = pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    from radharmony.evaluator.backbones.siglip2 import _SIGLIP2_HUB, _lowercase_text

    if not isinstance(hub.try_to_load_from_cache(_SIGLIP2_HUB, "tokenizer.json"), str):
        pytest.skip("SigLIP2 tokenizer not cached")
    processor = transformers.AutoProcessor.from_pretrained(_SIGLIP2_HUB)
    _lowercase_text(processor)
    a = processor(text=["a chest X-ray showing Cardiomegaly"], padding="max_length")["input_ids"]
    b = processor(text=["a chest x-ray showing cardiomegaly"], padding="max_length")["input_ids"]
    assert list(a[0]) == list(b[0])


# --------------------------------------------------------------------------
# chexagent_gen and maira2_gen: one PNG per source image (V-B-3), and the
# prompt argument (V-B-4). A fake transformers module replaces the models.
# --------------------------------------------------------------------------
class _FakeTokenizer:
    def __init__(self):
        self.texts: list[str] = []

    def from_list_format(self, items):
        self.texts.extend(i["text"] for i in items if "text" in i)
        return "query"

    def apply_chat_template(self, conv, **kwargs):
        return torch.zeros(1, 3, dtype=torch.long)

    def decode(self, ids, **kwargs):
        return "text"


class _FakeModel(torch.nn.Module):
    device = torch.device("cpu")

    def to(self, *args, **kwargs):
        return self

    def generate(self, ids, **kwargs):
        return torch.zeros(1, 5, dtype=torch.long)


@pytest.fixture
def fake_transformers(monkeypatch):
    import sys
    import types

    tok = _FakeTokenizer()
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = type("T", (), {"from_pretrained": staticmethod(lambda *a, **k: tok)})
    fake.AutoModelForCausalLM = type(
        "M", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeModel())})
    fake.AutoProcessor = type("P", (), {"from_pretrained": staticmethod(lambda *a, **k: object())})
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return tok


@pytest.fixture
def same_basename(tmp_path):
    """Two different images with the same file name, as in CheXpert."""
    from PIL import Image

    paths = []
    for i, patient in enumerate(("patient1", "patient2")):
        d = tmp_path / patient / "study1"
        d.mkdir(parents=True)
        p = d / "view1_frontal.png"
        Image.fromarray(_gray(16 + 8 * i, 16, seed=i)).save(p)
        paths.append(str(p))
    return paths


@pytest.mark.parametrize("factory", ["make_chexagent_generator", "make_maira2_generator"])
def test_generator_png_per_source_image(fake_transformers, same_basename, tmp_path, factory):
    from PIL import Image

    from radharmony.evaluator import backbones

    transform, _ = getattr(backbones, factory)(device="cpu", tmp_dir=str(tmp_path / "png"))
    pngs = [transform({"img": p, "report": "r"})["img"] for p in same_basename]
    assert pngs[0] != pngs[1]
    for src, png in zip(same_basename, pngs, strict=True):
        assert Image.open(png).size == Image.open(src).size


def test_chexagent_generator_uses_a_custom_prompt(fake_transformers):
    from radharmony.evaluator.backbones import make_chexagent_generator

    _, generate = make_chexagent_generator(device="cpu", prompt="CUSTOM PROMPT")
    generate(["a.png"])
    assert fake_transformers.texts == ["CUSTOM PROMPT"]


def test_chexagent_generator_puts_the_model_on_device(fake_transformers, monkeypatch):
    """V-B-15: device="cuda:3" reaches the model. Before, device_map="auto" placed it, and device was not used."""
    import sys

    calls = []

    def from_pretrained(*args, **kwargs):
        calls.append(kwargs)
        return _FakeModel()

    monkeypatch.setattr(sys.modules["transformers"].AutoModelForCausalLM, "from_pretrained",
                        staticmethod(from_pretrained))
    from radharmony.evaluator.backbones import make_chexagent_generator

    make_chexagent_generator(device="cuda:3")
    assert calls[0]["device_map"] == "cuda:3"


def test_chexagent_generator_default_is_the_abcde_recipe(fake_transformers):
    from radharmony.evaluator.backbones import make_chexagent_generator

    _, generate = make_chexagent_generator(device="cpu")
    generate(["a.png"])
    assert len(fake_transformers.texts) == 6  # 5 anatomy calls and 1 impression call


def test_png_folder_of_a_generator_goes_at_exit_unless_the_user_gave_it(tmp_path):
    """Decided by Frank Li (2026-11-04): the recipe deletes the PNG folder that it made. A given folder stays."""
    import subprocess
    import sys

    code = ("import sys\nfrom radharmony.evaluator.backbones._utils import png_cache_dir\n"
            "made = png_cache_dir(None, 'chexagent_png_')\ngiven = png_cache_dir(sys.argv[1], 'x')\n"
            "open(made + '/a.png', 'w').close()\nopen(given + '/a.png', 'w').close()\nprint(made)\n")
    given = tmp_path / "pngs"
    out = subprocess.run([sys.executable, "-c", code, str(given)], capture_output=True, text=True, check=True)
    made = out.stdout.strip()
    assert made and not os.path.exists(made)
    assert (given / "a.png").is_file()
