"""Worked example: VLM report generation scored with RadEval.

Dataset-agnostic — point it at any RadHarmony dataset that yields reference
reports (built with ``output_report=True``). The only model-specific piece
is the generator factory; swap ``make_chexagent_generator`` for your own
``(transform, report_generator)`` to evaluate a different VLM.

Install:
    uv pip install -e ".[radeval,chexagent_gen]"

Run:
    .venv/bin/python notebooks/evaluator/report_generation_example.py \
        --base-image-dir /mnt/NAS4/datasets/.../mimic-cxr/ \
        --report-csv     /mnt/NAS4/datasets/.../cxr-study-list.csv.gz \
        --output-dir      /mnt/NAS4/projects/jgichoy/results/chexagent_reportgen \
        --max-samples 50          # smoke test; drop for a full run
"""

from __future__ import annotations

import argparse
import os

from radharmony.evaluator import ReportGenerationEvaluator
from radharmony.evaluator.metrics import FULL_METRICS, LIGHT_METRICS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="openi",
                   choices=["openi", "mimic", "emory"],
                   help="Reference-report source. openi = OpenI-IU-CXR "
                        "(report text in-sample); mimic = MIMIC-CXR DICOM "
                        "(needs --report-csv); emory = EmoryCXR v2 "
                        "(needs --report-csv + --metadata-csv).")
    p.add_argument("--base-image-dir", required=True)
    p.add_argument("--report-csv", default=None,
                   help="MIMIC: cxr-study-list.csv.gz. Emory: "
                        "EmoryCXR_v2_Report_*.csv (de-id report text).")
    p.add_argument("--metadata-csv", default=None,
                   help="Emory: EmoryCXR_v2_Metadata_*.csv (image mapping).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ref-section", default=None,
                   choices=["findings", "impression", "both", "full"],
                   help="Default: 'full' for openi (harmonizer pre-merges "
                        "findings+impression), 'findings' for mimic.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-bootstrap", type=int, default=0)
    p.add_argument("--light", action="store_true",
                   help="Use the no-LLM/no-API LIGHT_METRICS subset.")
    p.add_argument("--mode", default="generate",
                   choices=["generate", "both"],
                   help="generate = Stage A only (dump pairs parquet; the "
                        "CheXagent-2 / two-env path). both = generate+score "
                        "in one env (only if one env satisfies both).")
    p.add_argument("--pairs-parquet", default=None,
                   help="Stage A output (default: <output-dir>/pairs.parquet).")
    p.add_argument("--subset-parquet", default=None,
                   help="Parquet of IDs to restrict the cohort to (e.g. the "
                        "47,490-study Emory external-validation set: "
                        "phase_grounding/data/emory_llm_labels.parquet) for "
                        "apples-to-apples with the linear probe.")
    p.add_argument("--subset-col", default="study_id",
                   help="Column joining --subset-parquet to the harmonized df.")
    p.add_argument("--backbone", default="chexagent",
                   choices=["chexagent", "maira2"],
                   help="Which generative VLM to evaluate. chexagent = "
                        "StanfordAIMI/CheXagent-2-3b (transformers==4.40.0 "
                        "env). maira2 = microsoft/maira-2 (transformers>=4.46 "
                        "env). Each backbone has its own pyproject extra and "
                        "must run in its own venv — see the runbook scripts.")
    args = p.parse_args()

    # Lazy backbone import — each backbone's extra has incompatible
    # transformers pins, so only the one we'll actually call is imported.
    if args.backbone == "chexagent":
        from radharmony.evaluator.backbones import make_chexagent_generator
        transform, report_generator = make_chexagent_generator(device=args.device)
    elif args.backbone == "maira2":
        from radharmony.evaluator.backbones import make_maira2_generator
        transform, report_generator = make_maira2_generator(device=args.device)
    else:
        raise ValueError(f"unknown backbone: {args.backbone}")

    # Dataset-agnostic: any RadHarmony dataset with output_report=True works.
    # Each branch sets (class, kwargs) so a subset filter can rebuild it via
    # harmonized_df= (base.py skips harmonization when that is passed).
    if args.dataset == "openi":
        from radharmony.dataset import OpenICXRDataset as DSCls  # heavy import

        ds_kwargs = dict(
            base_image_dir=args.base_image_dir,  # root w/ ecgen-radiology/ + images/
            transform=transform,
            output_report=True,
        )
        ref_section = args.ref_section or "full"
    elif args.dataset == "emory":
        from radharmony.dataset import EmoryCXRDataset as DSCls  # heavy import

        ds_kwargs = dict(
            base_image_dir=args.base_image_dir,   # EmoryCXRv2 DEID_PNG root
            csv_path=args.metadata_csv,           # EmoryCXR_v2_Metadata_*.csv
            report_csv_path=args.report_csv,      # EmoryCXR_v2_Report_*.csv
            transform=transform,
            output_report=True,
        )
        ref_section = args.ref_section or "findings"
    else:  # mimic — DICOM variant only; MIMIC-CXR-JPG has no reports
        from radharmony.dataset import MIMICCXRDataset as DSCls

        ds_kwargs = dict(
            base_image_dir=args.base_image_dir,
            report_csv_path=args.report_csv,
            transform=transform,
            output_report=True,
        )
        ref_section = args.ref_section or "findings"

    ds = DSCls(**ds_kwargs)

    # Apples-to-apples: restrict to the same cohort as the linear-probe
    # external validation by filtering the harmonized df, then rebuilding.
    if args.subset_parquet:
        import pandas as pd

        keep = set(
            pd.read_parquet(args.subset_parquet)[args.subset_col].astype(str)
        )
        hdf = ds._get_harmonized_df()
        if args.subset_col not in hdf.columns:
            raise KeyError(
                f"{args.subset_col!r} not in harmonized df columns "
                f"{list(hdf.columns)} — pass a valid --subset-col."
            )
        before = len(hdf)
        hdf = hdf[hdf[args.subset_col].astype(str).isin(keep)].copy()
        print(f"subset: {before} → {len(hdf)} rows "
              f"({len(keep)} ids in {args.subset_parquet})")
        ds = DSCls(**ds_kwargs, harmonized_df=hdf)

    # Smoke runs: when --max-samples is set, prefilter the harmonized df to
    # rows whose image file exists on disk and truncate to N. This (a) makes
    # the dataloader actually iterate N items (otherwise it walks the full
    # cohort and only the inner evaluator break-counter caps generation),
    # and (b) tolerates a partial dataset mirror (NAS4 MIMIC has only a
    # sparse subset of patient dirs vs. cxr-record-list.csv.gz).
    # MIMIC-only: the directory layout / partial-mirror reasoning is
    # specific to MIMIC-CXR-V2 on NAS4. Emory's PNG mirror is complete and
    # uses empi_anon as the dir name (no p-prefix, no patient_id-keyed
    # filter equivalent here), so we skip the prefilter there.
    if args.dataset == "mimic" and args.max_samples is not None and args.max_samples > 0:
        import os as _os

        hdf = ds._get_harmonized_df()
        # Build the set of mirrored patient_ids via O(pXX) listdir calls,
        # then filter the harmonized df by patient_id — much cheaper than
        # scanning tens of thousands of os.path.exists over NFS.
        # MIMIC patient dirs are named p<subject_id> (e.g. p10000032) but
        # the harmonizer's patient_id column stores subject_id verbatim
        # (10000032). Strip the leading "p" when building the mirrored set.
        mirrored: set = set()
        try:
            for pxx in _os.listdir(args.base_image_dir):
                pxx_dir = _os.path.join(args.base_image_dir, pxx)
                if _os.path.isdir(pxx_dir):
                    try:
                        for name in _os.listdir(pxx_dir):
                            mirrored.add(name[1:] if name.startswith("p") else name)
                    except OSError:
                        pass
        except OSError as e:
            raise RuntimeError(
                f"cannot list base_image_dir={args.base_image_dir}: {e}"
            ) from e
        before = len(hdf)
        if "patient_id" in hdf.columns and mirrored:
            hdf = hdf[hdf["patient_id"].astype(str).isin(mirrored)].copy()
        print(f"smoke prefilter: {before} → {len(hdf)} rows by mirrored "
              f"patient_id set ({len(mirrored)} mirrored patients).")
        # Final per-row exists() to drop patients with missing individual
        # files; cap at max_samples.
        hdf["_exists"] = hdf["image_path"].apply(
            lambda p: _os.path.exists(_os.path.join(args.base_image_dir, str(p)))
        )
        hdf = hdf[hdf["_exists"]].drop(columns=["_exists"]).head(args.max_samples)
        if len(hdf) < args.max_samples:
            raise RuntimeError(
                f"prefilter: only {len(hdf)} rows with existing images under "
                f"{args.base_image_dir} (need {args.max_samples}). Mirror is "
                f"too sparse for this smoke size."
            )
        print(f"smoke prefilter: kept {len(hdf)} rows with on-disk images.")
        ds = DSCls(**ds_kwargs, harmonized_df=hdf)

    ev = ReportGenerationEvaluator(
        report_generator,
        dataset=ds,
        metrics=LIGHT_METRICS if args.light else FULL_METRICS,
        ref_section=ref_section,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
    )

    if args.mode == "generate":
        # Stage A of the two-stage workflow: generate + dump pairs only.
        # No RadEval import — runs in the CheXagent-2 env (transformers
        # ==4.40.0). Score the parquet with score_reports_radeval.py.
        pairs = args.pairs_parquet or os.path.join(
            args.output_dir, "pairs.parquet"
        )
        path = ev.generate_only(pairs)
        print(f"\nGenerated {ev.n_samples_} (id,ref,hyp) pairs → {path}")
        print("Stage B: python score_reports_radeval.py "
              f"--pairs-parquet {path} --output-dir {args.output_dir}"
              f"{' --light' if args.light else ''}")
        return

    # Single-env path (only when one env can satisfy both — not the
    # CheXagent-2 case): generate + score together.
    df = ev.evaluate()
    out = ev.save_results(df)
    print(f"\nScored {ev.n_samples_} report pairs.")
    print(f"Results written to {out}/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
