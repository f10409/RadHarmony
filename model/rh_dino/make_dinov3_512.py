"""Write a folder of the original DINOv3 ViT-B/16 with an input size of 512 (the reference row of the paper).

The folder has the original config.json and model.safetensors, and a preprocessor_config.json with the size
512 x 512 (the Hugging Face default is 224). Use it as --hub of evaluate.py, without --scaling: the reference
row uses the RadHarmony DINOv3 recipe with its min-max scaling.

The Hugging Face model is gated. Accept its license on the model page, and log in (``hf auth login``) first.

Usage (from model/rh_dino/):
    python make_dinov3_512.py OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from huggingface_hub import hf_hub_download

HUB = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="output folder")
    ap.add_argument("--size", type=int, default=512, help="input height and width (default 512)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for f in ("config.json", "model.safetensors"):
        shutil.copyfile(hf_hub_download(HUB, f), os.path.join(a.out_dir, f))
    with open(hf_hub_download(HUB, "preprocessor_config.json")) as f:
        pp = json.load(f)
    pp["size"] = {"height": a.size, "width": a.size}
    with open(os.path.join(a.out_dir, "preprocessor_config.json"), "w") as f:
        json.dump(pp, f, indent=2)
    print(f"{HUB} at {a.size} x {a.size} -> {a.out_dir}")


if __name__ == "__main__":
    main()
