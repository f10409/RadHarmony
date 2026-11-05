#!/usr/bin/env python
"""Download and unpack MR-RATE study zips for a chosen cohort.

The full repo is 8.11 TB and every study is one zip. So you download whole
studies, not series. The packaging turns off pattern filtering, and asking for
one sequence still gets the whole study. Choose how many studies, not which files.

MR-RATE ships as FOUR repos; --repo picks one:

    main   8.11 TB   defaced native-space volumes, brain + defacing masks
    nvseg   415 GB   NV-Segment-CTMR segmentations (132-region brain, whole-body)
    coreg  17.60 TB  every series registered to the study's T1w center series
    atlas  12.30 TB  everything registered to the MNI152 atlas

Sizing measured 2026-08-22 on real downloads: main ~55 MB/study (18 files, 6 series),
nvseg ~2.6 MB/study. So the 5,568-study test split is ~300 GB from main but only
~15 GB from nvseg -- the segmentations are cheap, the volumes are not.

Requires HF auth with access granted to the gated repo:
    hf auth login

Usage:
    python scripts/materialize_mrrate.py --base-dir /data/MR-RATE \\
        --split test --limit 20 --unpack
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    # MR-RATE ships as four repos. Filenames and unpack roots differ per repo, so the
    # mapping lives here rather than being guessed at the call site.
    ap.add_argument("--repo", default="main",
                    choices=["main", "nvseg", "coreg", "atlas"],
                    help="main 8.1TB volumes+masks | nvseg 415GB segmentations | "
                         "coreg 17.6TB co-registered | atlas 12.3TB MNI152-registered")
    ap.add_argument("--split", default=None, choices=["train", "val", "test"],
                    help="restrict to one split (default: all)")
    ap.add_argument("--studies", nargs="*", default=None,
                    help="explicit study_uids, overrides --split/--limit")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of studies (not used with --studies)")
    ap.add_argument("--unpack", action="store_true",
                    help="extract each zip into <base-dir>/mri/ and keep the zip")
    ap.add_argument("--delete-zip", action="store_true",
                    help="remove the zip after a successful unpack (halves disk use)")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                    help="skip studies whose img/ directory already exists (default: on)")
    a = ap.parse_args()

    splits_csv = os.path.join(a.base_dir, "meta", "splits.csv")
    if not os.path.isfile(splits_csv):
        sys.exit(f"missing {splits_csv} -- download the metadata CSVs first")
    sp = pd.read_csv(splits_csv)

    if a.studies:
        sel = sp[sp.study_uid.isin(set(a.studies))]
    else:
        sel = sp[sp.split == a.split] if a.split else sp
        if a.limit:
            sel = sel.head(a.limit)
    if sel.empty:
        sys.exit("no studies selected")

    # repo -> (hf repo id, filename suffix, unpack root, measured MB/study)
    REPOS = {
        "main":  ("Forithmus/MR-RATE",              "",              "mri",       55.0),
        "nvseg": ("Forithmus/MR-RATE-nvseg-ctmr",   "_nvseg-ctmr",   "mri_nvseg",  2.6),
        "coreg": ("Forithmus/MR-RATE-coreg",        "",              "mri_coreg", 179.0),
        "atlas": ("Forithmus/MR-RATE-atlas",        "",              "mri_atlas", 125.0),
    }
    repo_id, suffix, root_name, mb = REPOS[a.repo]
    est_gb = len(sel) * mb / 1000.0
    print(f"{len(sel)} studies from {repo_id}  (~{est_gb:.1f} GB at {mb:.0f} MB/study)",
          flush=True)

    zip_root = os.path.join(a.base_dir, "zips" if a.repo == "main" else f"zips_{a.repo}")
    out_root = os.path.join(a.base_dir, root_name)
    got = skipped = failed = 0

    for i, r in enumerate(sel.itertuples(), 1):
        dest_dir = os.path.join(out_root, r.batch_id, r.study_uid)
        probe = "img" if a.repo in ("main", "coreg", "atlas") else "seg"
        if a.skip_existing and os.path.isdir(os.path.join(dest_dir, probe)):
            skipped += 1
            continue
        rel = f"mri/{r.batch_id}/{r.study_uid}{suffix}.zip"
        try:  # Import here, so a run that only skips needs no huggingface_hub.
            from huggingface_hub import hf_hub_download
        except ImportError:
            sys.exit("huggingface_hub is missing: pip install huggingface_hub")
        try:
            zp = hf_hub_download(repo_id, rel, repo_type="dataset",
                                 local_dir=zip_root)
        except Exception as exc:                      # gated repo, network, missing
            print(f"  [fail] {rel}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            failed += 1
            continue
        got += 1
        if a.unpack:
            # The archive already contains <study_uid>/img/... so extract one level up.
            with zipfile.ZipFile(zp) as z:
                z.extractall(os.path.join(out_root, r.batch_id))
            if a.delete_zip:
                os.remove(zp)
        if i % 10 == 0 or i == len(sel):
            print(f"  [{i}/{len(sel)}] downloaded {got}, skipped {skipped}, failed {failed}",
                  flush=True)

    print(f"\ndone: {got} downloaded, {skipped} already present, {failed} failed")
    if got and not a.unpack:
        print("zips are under zips/ -- rerun with --unpack to extract, or the "
              "harmonizer will not find any image files")


if __name__ == "__main__":
    main()
