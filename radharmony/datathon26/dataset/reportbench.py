"""End-to-end bridge from raw studies to a datathon26 dataset via *reportbench*.

The datathon26 result datasets (:class:`EmbeddingResultsDataset`,
:class:`ReportResultsDataset`) score results the external **reportbench**
inference service has already written to disk. Getting those results is a manual
CLI dance (``rbclient.py prepare`` → ``submit`` → ``watch`` → per-study files).

This module wraps that dance so a caller can go straight from a harmonized frame
and the raw image root — *the same two inputs an ordinary RadHarmony dataset
takes* — to a ready dataset object:

    ds = DatathonEmbeddingDataset(
        harmonized_df=df,                 # one row per study (labels + report)
        base_image_dir="/data/mimic",     # where the raw images live
        client=ReportBenchClient(
            rbclient_path="/mnt/NAS4/.../skill/rbclient.py",
            data_dir="/mnt/NAS4/.../data_deposition/<team_id>",
            api_key="rb_...",
        ),
        model="model-a",
    )
    LinearProbeEvaluator = ...            # score ds as usual

Under the hood it (1) stages the studies referenced by ``harmonized_df`` into the
per-study folder layout the service expects, (2) drives ``rbclient.py`` to submit
the job and wait for it, then (3) points the matching datathon26 dataset at the
returned results folder.

The ``rbclient.py`` CLI itself is treated as an opaque tool driven by subprocess
(per the datathon ``INSTRUCTIONS.md``): ``config`` / ``check`` / ``prepare`` /
``submit --model <m> --task <t> --input <run>`` / ``watch`` / ``results``. Every
path (the client script, the Python interpreter, the data / staging / results
directories) is configurable so the wrapper survives details not visible here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Sequence

import pandas as pd
import torch

from .embedding_dataset import EmbeddingResultsDataset
from .report_dataset import ReportResultsDataset


class ReportBenchClient:
    """Thin subprocess wrapper around the datathon ``rbclient.py`` CLI.

    Each method maps to one ``rbclient.py`` sub-command. Output is streamed to
    this process's stdout/stderr so long-running steps (``watch``) show their
    progress live. A non-zero exit raises :class:`subprocess.CalledProcessError`.

    Args:
        rbclient_path: Path to ``rbclient.py`` (on the datathon server, typically
            ``/mnt/NAS4/projects/bkhosra/sharing/datathon/skill/rbclient.py``).
        data_dir: Your team data folder — the ``--data-dir`` the CLI writes
            submissions and results into. Results for run ``<name>`` land in
            ``<data_dir>/<name>/<study>/`` (see :attr:`results_dir`).
        api_key: Team API key (``rb_...``). When given, :meth:`config` is run
            once before the first job to persist the key + data dir.
        python_exe: Interpreter used to launch the CLI (default ``python3``).
        cwd: Working directory for the CLI (defaults to ``rbclient.py``'s dir,
            matching the documented ``cd .../skill`` usage).
        env: Extra environment variables to overlay on the child process.
    """

    def __init__(
        self,
        *,
        rbclient_path: str,
        data_dir: str,
        api_key: str | None = None,
        python_exe: str = "python3",
        cwd: str | None = None,
        env: dict | None = None,
    ):
        self.rbclient_path = os.path.abspath(rbclient_path)
        self.data_dir = os.path.abspath(data_dir)
        self.api_key = api_key
        self.python_exe = python_exe
        self.cwd = cwd or os.path.dirname(self.rbclient_path)
        self.env = env
        self._configured = False

    # ── low-level ────────────────────────────────────────────────────────
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        cmd = [self.python_exe, self.rbclient_path, *args]
        env = None
        if self.env:
            env = {**os.environ, **self.env}
        print("[reportbench] $", " ".join(cmd))
        return subprocess.run(cmd, cwd=self.cwd, env=env, check=True)

    # ── CLI sub-commands ───────────────────────────────────────────────────
    def config(self) -> None:
        """Persist the API key + data dir (one-time; safe to repeat)."""
        if self.api_key is None:
            raise ValueError("api_key is required to run `rbclient.py config`.")
        self._run("config", "--api-key", self.api_key, "--data-dir", self.data_dir)
        self._configured = True

    def ensure_configured(self) -> None:
        if not self._configured and self.api_key is not None:
            self.config()

    def check(self) -> None:
        """Confirm connectivity to the service."""
        self._run("check")

    def models(self) -> None:
        """List available models + tasks."""
        self._run("models")

    def prepare(self, from_dir: str, name: str) -> None:
        """Copy the staged study folders in ``from_dir`` into the run ``name``."""
        self._run("prepare", "--from", os.path.abspath(from_dir), "--name", name)

    def submit(self, model: str, task: str, run_name: str) -> None:
        """Submit run ``run_name`` to ``model`` for ``task`` (``report``/``embeddings``)."""
        self._run("submit", "--model", model, "--task", task, "--input", run_name)

    def watch(self) -> None:
        """Block until the in-flight job completes."""
        self._run("watch")

    def results(self) -> None:
        """Print the results summary."""
        self._run("results", "--print")

    def results_dir(self, run_name: str) -> str:
        """Directory the service writes per-study results into for ``run_name``."""
        return os.path.join(self.data_dir, run_name)


def _stage_studies(
    harmonized_df: pd.DataFrame,
    base_image_dir: str,
    staging_dir: str,
    *,
    image_col: str,
    study_col: str,
    indication_col: str | None,
    link: bool,
) -> int:
    """Lay the studies referenced by ``harmonized_df`` out for ``rbclient prepare``.

    Builds ``<staging_dir>/<study_id>/<view files>`` (+ optional ``text.txt``),
    grouping every image row of a study into that study's folder. Returns the
    number of study folders created.
    """
    if study_col not in harmonized_df.columns:
        raise KeyError(f"harmonized_df has no '{study_col}' column.")
    if image_col not in harmonized_df.columns:
        raise KeyError(f"harmonized_df has no '{image_col}' column.")

    os.makedirs(staging_dir, exist_ok=True)
    n_studies = 0
    for study_id, group in harmonized_df.groupby(study_col, sort=False):
        study_dir = os.path.join(staging_dir, str(study_id))
        os.makedirs(study_dir, exist_ok=True)
        seen: dict[str, int] = {}
        for _, row in group.iterrows():
            rel = str(row[image_col])
            src = os.path.join(base_image_dir, rel) if base_image_dir else rel
            fname = os.path.basename(rel)
            # Disambiguate colliding basenames within one study (e.g. two views
            # both named image.dcm from different subfolders).
            if fname in seen:
                seen[fname] += 1
                stem, ext = os.path.splitext(fname)
                fname = f"{stem}_{seen[fname]}{ext}"
            else:
                seen[fname] = 0
            dst = os.path.join(study_dir, fname)
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            if link:
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
        if indication_col and indication_col in group.columns:
            text = str(group.iloc[0][indication_col] or "").strip()
            if text and text.lower() != "nan":
                with open(os.path.join(study_dir, "text.txt"), "w") as f:
                    f.write(text)
        n_studies += 1
    return n_studies


def _run_reportbench(
    harmonized_df: pd.DataFrame,
    base_image_dir: str,
    client: ReportBenchClient,
    *,
    model: str,
    task: str,
    run_name: str,
    staging_dir: str | None,
    results_dir: str | None,
    image_col: str,
    study_col: str,
    indication_col: str | None,
    link: bool,
    check: bool,
    skip_submit: bool,
    result_filename: str,
) -> str:
    """Stage → submit → watch, and return the results directory to score.

    When ``skip_submit`` is set the staging/submit/watch steps are skipped and
    the existing ``results_dir`` is reused (re-scoring a finished run).
    """
    resolved_results = results_dir or client.results_dir(run_name)

    if not skip_submit:
        staging = staging_dir or os.path.join(client.data_dir, ".staging", run_name)
        n = _stage_studies(
            harmonized_df,
            base_image_dir,
            staging,
            image_col=image_col,
            study_col=study_col,
            indication_col=indication_col,
            link=link,
        )
        print(f"[reportbench] staged {n} studies into {staging}")

        client.ensure_configured()
        if check:
            client.check()
        client.prepare(staging, run_name)
        client.submit(model, task, run_name)
        client.watch()
        client.results()

    _verify_results(harmonized_df, resolved_results, study_col, result_filename)
    return resolved_results


def _verify_results(
    harmonized_df: pd.DataFrame,
    results_dir: str,
    study_col: str,
    result_filename: str,
) -> None:
    """Fail early (with the missing study ids) if the join won't resolve."""
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}. Pass results_dir= "
            "if rbclient.py wrote results somewhere else."
        )
    missing: list[str] = []
    for study_id in harmonized_df[study_col].astype(str).unique():
        if not os.path.isfile(os.path.join(results_dir, study_id, result_filename)):
            missing.append(study_id)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)}/{harmonized_df[study_col].nunique()} studies are "
            f"missing {result_filename} under {results_dir} (e.g. {preview}). "
            "Check that rbclient.py preserved the study-folder names and that the "
            "job finished; override with results_dir= if needed."
        )


def DatathonEmbeddingDataset(
    harmonized_df: pd.DataFrame,
    base_image_dir: str,
    *,
    client: ReportBenchClient,
    model: str,
    run_name: str = "run_embeddings",
    task: str = "embeddings",
    staging_dir: str | None = None,
    results_dir: str | None = None,
    image_col: str = "image_path",
    study_col: str = "study_id",
    indication_col: str | None = None,
    link: bool = True,
    check: bool = False,
    skip_submit: bool = False,
    label_cols: Sequence[str] | None = None,
    cache_dir: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> EmbeddingResultsDataset:
    """Run the embeddings task through reportbench and return a ready dataset.

    Takes the same two inputs as an ordinary RadHarmony dataset — a harmonized
    frame and the raw image root — stages the studies, submits the *embeddings*
    task via :class:`ReportBenchClient`, waits for it, then returns an
    :class:`EmbeddingResultsDataset` pointed at the ``<study>/embedding.npy``
    results.

    (This is a factory function, not a class: it does the submission work up
    front and hands back the underlying :class:`EmbeddingResultsDataset`.)

    Args:
        harmonized_df: One row per study, with ``study_col`` / ``image_col`` and
            the one-hot label columns (as produced by ``sample_data.ipynb``).
        base_image_dir: Root prepended to ``image_col`` to locate the raw images.
        client: Configured :class:`ReportBenchClient`.
        model: reportbench model id (e.g. ``"model-a"``). See ``client.models()``.
        run_name: Submission/run name; results land in ``<data_dir>/<run_name>``.
        task: reportbench task name for embeddings (default ``"embeddings"``).
        staging_dir: Where to build the per-study submission layout (default
            ``<data_dir>/.staging/<run_name>``).
        results_dir: Override the results directory (default
            ``client.results_dir(run_name)``).
        image_col, study_col: Column names in ``harmonized_df``.
        indication_col: Optional column whose text is written as each study's
            ``text.txt`` indication.
        link: Symlink staged images (default) instead of copying.
        check: Run ``rbclient.py check`` before submitting.
        skip_submit: Skip staging/submit/watch and reuse an existing
            ``results_dir`` (re-score a finished run).
        label_cols, cache_dir, dtype: Forwarded to :class:`EmbeddingResultsDataset`.
    """
    resolved = _run_reportbench(
        harmonized_df,
        base_image_dir,
        client,
        model=model,
        task=task,
        run_name=run_name,
        staging_dir=staging_dir,
        results_dir=results_dir,
        image_col=image_col,
        study_col=study_col,
        indication_col=indication_col,
        link=link,
        check=check,
        skip_submit=skip_submit,
        result_filename="embedding.npy",
    )
    return EmbeddingResultsDataset(
        resolved,
        harmonized_df=harmonized_df,
        label_cols=list(label_cols) if label_cols is not None else None,
        cache_dir=cache_dir,
        dtype=dtype,
    )


def DatathonReportDataset(
    harmonized_df: pd.DataFrame,
    base_image_dir: str,
    *,
    client: ReportBenchClient,
    model: str,
    run_name: str = "run_report",
    task: str = "report",
    staging_dir: str | None = None,
    results_dir: str | None = None,
    image_col: str = "image_path",
    study_col: str = "study_id",
    indication_col: str | None = None,
    link: bool = True,
    check: bool = False,
    skip_submit: bool = False,
    cache_dir: str | None = None,
) -> ReportResultsDataset:
    """Run the report task through reportbench and return a ready dataset.

    The report-generation counterpart of :func:`DatathonEmbeddingDataset` (also a
    factory function, not a class): stages studies, submits the *report* task,
    waits, then returns a :class:`ReportResultsDataset` pointed at the
    ``<study>/report.txt`` results (its inline ``report`` column supplies the
    reference text). See that function for the shared arguments.
    """
    resolved = _run_reportbench(
        harmonized_df,
        base_image_dir,
        client,
        model=model,
        task=task,
        run_name=run_name,
        staging_dir=staging_dir,
        results_dir=results_dir,
        image_col=image_col,
        study_col=study_col,
        indication_col=indication_col,
        link=link,
        check=check,
        skip_submit=skip_submit,
        result_filename="report.txt",
    )
    return ReportResultsDataset(
        resolved,
        harmonized_df=harmonized_df,
        cache_dir=cache_dir,
    )
