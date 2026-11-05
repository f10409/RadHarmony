"""EVA-X gives the feature of its authors (V-B-1, D-16).

The EVA-X fine-tune scripts use ``--use_mean_pooling``: the mean of the patch
tokens, then the final norm ``fc_norm``. The test runs where the ``eva_x`` extra
is installed and the base weights are in the Hugging Face cache.

Run with::

    .venv/bin/python -m pytest tests/test_eva_x.py -v
"""

import pytest
import torch

pytest.importorskip("timm")
huggingface_hub = pytest.importorskip("huggingface_hub")

WEIGHTS = huggingface_hub.try_to_load_from_cache(
    "MapleF/eva_x", "eva_x_base_patch16_merged520k_mim.pt")


@pytest.mark.skipif(not isinstance(WEIGHTS, str), reason="EVA-X base weights not cached")
def test_feature_is_mean_of_patch_tokens_then_fc_norm():
    """The feature is fc_norm(mean of patch tokens), and fc_norm has the pretrained final norm."""
    from radharmony.evaluator.backbones import make_eva_x

    _, encoder = make_eva_x(device="cpu", variant="base", dtype=torch.float32)
    model = encoder.model
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)["module"]
    torch.testing.assert_close(model.fc_norm.weight, checkpoint["norm.weight"].float())
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        got = encoder(x)
        tokens = model.forward_features(x)
        expected = model.fc_norm(tokens[:, model.num_prefix_tokens:].mean(dim=1))
    assert got.shape == (2, 768)
    torch.testing.assert_close(got, expected)
