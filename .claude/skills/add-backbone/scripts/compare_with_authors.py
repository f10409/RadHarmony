"""Compare a backbone recipe with the authors' own pipeline (J-05, D-16).

The authors file is a Python file with one function, ``load(device)``. It returns 2 or 3 functions:

- ``prepare(path) -> Tensor[C, H, W]``: the authors' input steps for one image file.
- ``feature(x) -> Tensor[B, D]``: the authors' model and pooled token, on a batch from ``prepare``.
- ``patches(x) -> Tensor[B, D, G, G]`` (only for ``--seg``): the authors' patch tokens as a grid.

With ``--seg``, the script builds the recipe in segmentation mode and compares the patch features with
``patches`` in place of the pooled feature. It does not check the mask steps.

For each image, the script prints 3 cosine values:

- input: the recipe transform against ``prepare``. It checks the input size, the crop, the mean and the std.
- token: the recipe encoder against ``feature``, on the same authors' input. It checks the pooled token.
- end: the recipe (transform and encoder) against the authors (``prepare`` and ``feature``).

It exits with 1 when a shape differs or a cosine is below ``--min-cos``.

The images come from ``--images``, or from ``--dataset <key>``: the first ``--n`` images of a dataset. The
folder of ``--dataset`` comes from the data paths file (``configs/data_paths.json``, or the file that
``RADHARMONY_DATA_PATHS`` names), read with ``scripts/dataset_paths.py``.

Usage::

    CUDA_VISIBLE_DEVICES=0 $RADHARMONY_MODEL_PYTHON .claude/skills/add-backbone/scripts/compare_with_authors.py \\
        --recipe make_raddino --authors /tmp/authors_raddino.py --dataset montgomery_cxr --n 4
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
# The repo goes first, so that the model venv imports this checkout and not another install.
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import torch  # noqa: E402


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _dataset_images(key: str, n: int) -> list[str]:
    """Return the first ``n`` image files of a dataset, from the data paths file."""
    from dataset_paths import MISSING, paths

    from radharmony.registry import resolve_dataset

    args = paths(key, missing_ok=True)
    if MISSING in args:
        raise SystemExit(f"{args[MISSING]} Or give --images.")
    if "base_image_dir" not in args:
        raise SystemExit(f"{key!r} has no base_image_dir in the data paths file. Give --images instead.")
    df = resolve_dataset(key)(**args, cache_dir=None).get_harmonized_df()
    return [str(Path(args["base_image_dir"]) / p) for p in df["image_path"].head(n)]


def _load_authors(path: str):
    spec = importlib.util.spec_from_file_location("authors_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", required=True, help="factory name in radharmony.evaluator.backbones")
    ap.add_argument("--authors", required=True,
                    help="Python file with load(device) -> (prepare, feature) or (prepare, feature, patches)")
    ap.add_argument("--recipe-kwargs", default="{}", help='JSON, for example \'{"checkpoint_path": "/data/x.pth"}\'')
    ap.add_argument("--images", nargs="*", help="image files, for example 4 chest X-rays")
    ap.add_argument("--dataset", help="a registry key in the data paths file")
    ap.add_argument("--n", type=int, default=4, help="number of images from --dataset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--seg", action="store_true", help="compare the patch features of segmentation mode")
    args = ap.parse_args()
    if not args.images and not args.dataset:
        ap.error("give --images or --dataset")
    images = args.images or _dataset_images(args.dataset, args.n)
    if not images:
        raise SystemExit("no images to compare")

    import radharmony.evaluator.backbones as backbones

    factory = getattr(backbones, args.recipe)
    kwargs = json.loads(args.recipe_kwargs)
    if "dtype" in inspect.signature(factory).parameters:
        kwargs.setdefault("dtype", torch.float32)  # float32 on both sides: no bfloat16 rounding
    if args.seg:
        kwargs["output_keys"] = {"img", "mask"}
    out = factory(device=args.device, **kwargs)
    transform, encoder = out[0], out[1].eval()
    fns = _load_authors(args.authors)(args.device)
    if args.seg and len(fns) < 3:
        raise SystemExit("--seg needs load(device) to return (prepare, feature, patches)")
    prepare, feature = fns[0], (fns[2] if args.seg else fns[1])

    ok = True
    print(f"{'image':<32} {'recipe input':<16} {'authors input':<16} {'input':>8} {'token':>8} {'end':>8}")
    for path in images:
        # In segmentation mode the transform needs a mask. The image file stands in for it (not checked).
        sample = {"img": path, "mask": path} if args.seg else {"img": path, "cls": torch.zeros(1)}
        ours = transform(sample)["img"].float()
        theirs = prepare(path).float()
        same_shape = ours.shape == theirs.shape
        with torch.no_grad():
            x = theirs[None].to(args.device)
            f_auth = feature(x).float().cpu()
            f_tok = encoder(x).float().cpu()
            f_end = encoder(ours[None].to(args.device)).float().cpu() if same_shape else None
        c_in = _cos(ours, theirs) if same_shape else float("nan")
        c_tok = _cos(f_tok, f_auth) if f_tok.shape == f_auth.shape else float("nan")
        c_end = _cos(f_end, f_auth) if f_end is not None and f_end.shape == f_auth.shape else float("nan")
        # NaN fails too: a shape that differs is a mismatch.
        ok &= all(c >= args.min_cos for c in (c_in, c_tok, c_end))
        print(f"{Path(path).name[:32]:<32} {str(tuple(ours.shape)):<16} {str(tuple(theirs.shape)):<16} "
              f"{c_in:8.5f} {c_tok:8.5f} {c_end:8.5f}")
    print(f"feature size: recipe {tuple(f_tok.shape[1:])}, authors {tuple(f_auth.shape[1:])}")
    print("PASS" if ok else f"FAIL: a shape differs or a cosine is below {args.min_cos}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
