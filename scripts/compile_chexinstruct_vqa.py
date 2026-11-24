#!/usr/bin/env python
"""Write ``chexinstruct_vqa.json`` for ``ChexInstructVQAHarmonizer`` from the CheXagent processors.

It runs ``MIMICCXRVQAProcessor`` (open and closed questions) and ``MIMICDiffVQAProcessor``
(difference questions) with no change, and writes one JSON list with one row per sample.
Each row gets ``split``: ``train``, ``val`` or ``test``.

The processors use ``random`` with no seed. ``MIMICCXRVQAProcessor`` shuffles its train list, and the
closed-question template picks the choice letters and the option order in ``qa_pair``. ``unique_id`` (the
harmonizer's ``question_id``) of a MIMIC-CXR-VQA train row is its index after the shuffle. So the script
seeds ``random`` first (``--seed``, default 0), and 2 runs with the same seed give the same file.

The processors are not in this repo, because their license (CC BY-NC-ND 4.0) allows no changed
copies. Copy them from https://github.com/Stanford-AIMI/CheXagent, commit
e4f31e6e517f442620fe816f90c846cbcb28b746, folder ``data_chexinstruct/dataset_processors/``:
``base_processor.py``, ``templates.py``, ``utils.py``, ``mimic_cxr_vqa.py`` and ``mimic_diff_vqa.py``.
Do not copy its ``__init__.py``: it imports all 28 processors, so the import fails with only these 5 files. The processors read ``data/...``
paths, so the script runs in the vendor folder. This script is our own code (first written as
``vendor/compile_vqa.py`` in the data folder). It is not derived from CheXagent code.

Layout:

    <vendor-dir>/
      dataset_processors/   the 5 CheXagent files
      data/                 mimic-cxr/, mimic-cxr-vqa/, mimic-diff-vqa/ (or a link to them)

Usage:
    python scripts/compile_chexinstruct_vqa.py --vendor-dir /data/CheXinstruct/vendor
    python scripts/compile_chexinstruct_vqa.py --vendor-dir ... --out /data/CheXinstruct/compiled/chexinstruct_vqa.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


def _flatten(dataset: dict) -> list[dict]:
    """Return the samples of the 3 splits in one list, each with its ``split``."""
    samples = []
    for split in ("train", "val", "test"):
        for sample in dataset[split]:
            samples.append({**sample, "split": split})
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vendor-dir", required=True, help="Folder with dataset_processors/ and data/.")
    ap.add_argument("--out", default=None,
                    help="Output JSON. Default: <vendor-dir>/../compiled/chexinstruct_vqa.json.")
    ap.add_argument("--seed", type=int, default=0, help="Seed of random, for the same file in each run.")
    a = ap.parse_args()

    vendor = os.path.abspath(a.vendor_dir)
    # realpath: with a linked vendor folder, ".." is the folder above the real one, as for the processors.
    out = os.path.abspath(a.out or os.path.join(os.path.realpath(vendor), "..", "compiled", "chexinstruct_vqa.json"))
    # The processors read data/... relative paths and import dataset_processors from the vendor folder.
    os.chdir(vendor)
    sys.path.insert(0, vendor)
    from dataset_processors.mimic_cxr_vqa import MIMICCXRVQAProcessor
    from dataset_processors.mimic_diff_vqa import MIMICDiffVQAProcessor

    random.seed(a.seed)
    cxr_vqa = MIMICCXRVQAProcessor()
    diff_vqa = MIMICDiffVQAProcessor()
    samples = (_flatten(cxr_vqa.create_open_ended_vqa()) + _flatten(cxr_vqa.create_close_ended_vqa())
               + _flatten(diff_vqa.create_difference_vqa()))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(samples, f)
    print(f"Wrote {len(samples)} samples to {out}")


if __name__ == "__main__":
    main()
