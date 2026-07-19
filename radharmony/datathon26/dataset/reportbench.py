"""End-to-end bridge from raw studies to a datathon26 dataset via *reportbench*.

The datathon26 result datasets (:class:`EmbeddingResultsDataset`,
:class:`ReportResultsDataset`) score results the external **reportbench**
inference service has already written to disk. Getting those results is a manual
CLI dance (``rbclient.py prepare`` → ``submit`` → ``watch`` → per-study files).

This module wraps that dance so a caller can go straight from a harmonized frame
and the raw image root — *the same two inputs an ordinary RadHarmony dataset
takes* — to a ready dataset object:

    ds = DatathonEmbeddingDataset(
        harmonized_df=df,                 # one row per image (labels + report)
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

Compatibility with the live reportbench (as of the study-keyed manifest + ``.npz``
embedding format):

* **Tasks** are ``report`` and ``embed`` (not ``embeddings``).
* **Indication** is supplied through the job ``manifest.json`` (study-keyed), never
  a per-study ``text.txt`` — the file mechanism was removed server-side.
* **Manifest** schema is
  ``{"studies": {"<study_id>": {"images": [{"path", "view"}], "indication"?}}}``
  where ``path`` is a bare filename inside the study folder. It is written for the
  *report* task only when an ``indication_col`` is given (so an indication can be
  attached); the *embed* task submits without a manifest (views/indications don't
  affect embeddings, and a manifest would only add rejection surface).
* **Outputs** land under ``<data_dir>/<run>/_reportbench_out/<model>/<study>/`` as
  ``<model>_report.txt`` (per study) and ``<model>_<stem>.npz`` (per image; an
  archive with ``global`` / ``patches``).

The ``rbclient.py`` CLI itself is treated as an opaque tool driven by subprocess
(per the datathon ``INSTRUCTIONS.md``): ``config`` / ``check`` / ``prepare`` /
``submit --model <m> --task <t> --input <run>`` / ``watch`` / ``results``. Every
path (the client script, the Python interpreter, the data / staging / results
directories) is configurable so the wrapper survives details not visible here.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from typing import Callable, Sequence

import pandas as pd
import torch

from .embedding_dataset import EmbeddingResultsDataset, viewwise_embedding_path
from .report_dataset import ReportResultsDataset, perstudy_report_path
from .seg_dataset import PatchSegResultsDataset

#: Raw ``view_position`` strings → the manifest's allowed labels (ap/pa/lateral/ll).
_VIEW_ALIASES = {
    "ap": "ap", "pa": "pa", "lateral": "lateral", "ll": "ll",
    "lat": "lateral", "frontal": "pa", "front": "pa",
}


class ReportBenchClient:
    """Thin subprocess wrapper around the datathon ``rbclient.py`` CLI.

    Each method maps to one ``rbclient.py`` sub-command. Output is streamed to
    this process's stdout/stderr so long-running steps (``watch``) show their
    progress live. A non-zero exit raises :class:`subprocess.CalledProcessError`.

    Args:
        rbclient_path: Path to ``rbclient.py`` (on the datathon server, typically
            ``/mnt/NAS4/projects/bkhosra/sharing/datathon/skill/rbclient.py``).
        data_dir: Your team data folder — the ``--data-dir`` the CLI writes
            submissions and results into. The submitted run ``<name>`` lands in
            ``<data_dir>/<name>/`` and the service writes outputs under
            ``<data_dir>/<name>/_reportbench_out/<model>/`` (see
            :meth:`output_dir`).
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
        world_readable: bool = True,
    ):
        self.rbclient_path = os.path.abspath(rbclient_path)
        self.data_dir = os.path.abspath(data_dir)
        self.api_key = api_key
        self.python_exe = python_exe
        self.cwd = cwd or os.path.dirname(self.rbclient_path)
        self.env = env
        self.world_readable = world_readable
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
        """Copy the staged study folders in ``from_dir`` into the run ``name``.

        Called without ``--link`` so ``rbclient.py`` writes real file copies into
        the run folder — the job manifest resolves image paths *inside* the run
        folder, which the service requires (a symlink whose target escapes the
        folder is rejected).
        """
        self._run("prepare", "--from", os.path.abspath(from_dir), "--name", name)

    def make_world_readable(self, run_name: str) -> None:
        """Add ``o+rX`` to the prepared run tree (and ``o+x`` to ``data_dir``).

        ``rbclient.py prepare`` copies studies with the *prepare subprocess's*
        umask — on shared systems commonly ``007``, which leaves the run and
        study directories ``o---``. The reportbench service runs as a different
        user and reads the batch over NFS, so those directories are untraversable
        and ``submit`` fails with an opaque HTTP 500. Making the tree
        world-readable is deterministic (unlike umask propagation into the
        subprocess) and safe for the anonymized studies deposited here. Owner is
        unchanged; only the "other" read/traverse bits are added.
        """
        def _add(path: str, bits: int) -> None:
            try:
                os.chmod(path, os.stat(path).st_mode | bits)
            except OSError:
                pass  # not owner / transient NFS error — leave as-is

        _add(self.data_dir, stat.S_IXOTH)  # traverse into the batch root
        root = self.run_dir(run_name)
        _add(root, stat.S_IROTH | stat.S_IXOTH)
        for dirpath, dirnames, filenames in os.walk(root):
            for d in dirnames:
                _add(os.path.join(dirpath, d), stat.S_IROTH | stat.S_IXOTH)
            for f in filenames:
                _add(os.path.join(dirpath, f), stat.S_IROTH)

    def submit(self, model: str, task: str, run_name: str) -> None:
        """Submit run ``run_name`` to ``model`` for ``task`` (``report``/``embed``)."""
        self._run("submit", "--model", model, "--task", task, "--input", run_name)

    def watch(self) -> None:
        """Block until the in-flight job completes."""
        self._run("watch")

    def results(self) -> None:
        """Print the results summary."""
        self._run("results", "--print")

    def run_dir(self, run_name: str) -> str:
        """The submitted run/input folder: ``<data_dir>/<run_name>``."""
        return os.path.join(self.data_dir, run_name)

    # Back-compat alias — earlier code called this ``results_dir``.
    results_dir = run_dir

    def output_dir(self, run_name: str, model: str) -> str:
        """Where the service writes per-study outputs for ``run_name`` + ``model``.

        ``<data_dir>/<run_name>/_reportbench_out/<model>``.
        """
        return os.path.join(self.data_dir, run_name, "_reportbench_out", model)

    def ensure_output_writable(self, run_name: str, mode: int = 0o777) -> None:
        """Pre-create ``<run>/_reportbench_out`` so the service can write results.

        The service writes to ``<input>/_reportbench_out/<model>`` and often runs
        as a **different user** than the one staging the batch. If the run folder
        isn't writable by that user, its output ``mkdir`` fails and submit returns
        403 (older servers: a bare 500). We create the output root ahead of time
        and open its permissions (default world-writable — the batch already lives
        under an unguessable, non-listable team path) so the service's per-model
        ``mkdir`` succeeds regardless of who owns the run folder. Best-effort:
        chmod failures (e.g. we don't own the dir) are ignored.
        """
        run_dir = self.run_dir(run_name)
        out_root = os.path.join(run_dir, "_reportbench_out")
        os.makedirs(out_root, exist_ok=True)
        for d in (run_dir, out_root):
            try:
                os.chmod(d, mode)
            except OSError:
                pass


def _study_view_labels(view_positions: Sequence) -> list[str]:
    """Map a study's ``view_position`` values to manifest labels (ap/pa/lateral/ll).

    Guarantees a valid labelling for ≤2 images (≤1 frontal + ≤1 lateral): if any
    value is unmappable, or the mapped labels would put two images in the same
    frontal/lateral group, fall back to positional labels (first → ``pa``, second
    → ``lateral``). Report models ignore the labels; only the view-aware model
    (model-e) uses them, so a best-effort positional guess is the safe default.
    """
    labels = [
        _VIEW_ALIASES.get(str(vp).strip().lower()) if vp is not None and str(vp).strip() else None
        for vp in view_positions
    ]

    def _group(lbl):
        if lbl in ("ap", "pa"):
            return "frontal"
        if lbl in ("lateral", "ll"):
            return "lateral"
        return None

    groups = [_group(lbl) for lbl in labels]
    if any(lbl is None for lbl in labels) or groups.count("frontal") > 1 or groups.count("lateral") > 1:
        return ["pa" if i == 0 else "lateral" for i in range(len(view_positions))]
    return labels  # type: ignore[return-value]


def _stage_studies(
    harmonized_df: pd.DataFrame,
    base_image_dir: str,
    staging_dir: str,
    *,
    image_col: str,
    study_col: str,
    view_col: str | None,
    indication_col: str | None,
    link: bool,
) -> dict:
    """Lay the studies referenced by ``harmonized_df`` out for ``rbclient prepare``.

    Builds ``<staging_dir>/<study_id>/<view files>``, grouping every image row of
    a study into that study's folder, and returns the study-keyed **manifest**
    ``{"studies": {"<study_id>": {"images": [{"path", "view"}], "indication"?}}}``
    describing exactly what was staged. The caller decides whether to write the
    manifest (see :func:`_run_reportbench`).
    """
    if study_col not in harmonized_df.columns:
        raise KeyError(f"harmonized_df has no '{study_col}' column.")
    if image_col not in harmonized_df.columns:
        raise KeyError(f"harmonized_df has no '{image_col}' column.")

    os.makedirs(staging_dir, exist_ok=True)
    manifest: dict = {"studies": {}}
    for study_id, group in harmonized_df.groupby(study_col, sort=False):
        study_dir = os.path.join(staging_dir, str(study_id))
        os.makedirs(study_dir, exist_ok=True)
        seen: dict[str, int] = {}
        staged: list[tuple[str, object]] = []  # (filename in the study folder, view_position)
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
            if not (os.path.exists(dst) or os.path.islink(dst)):
                if link:
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
            vp = row[view_col] if view_col and view_col in group.columns else None
            staged.append((fname, vp))

        views = _study_view_labels([vp for _, vp in staged])
        entry: dict = {"images": [
            {"path": fname, "view": view} for (fname, _), view in zip(staged, views)
        ]}
        if indication_col and indication_col in group.columns:
            text = str(group.iloc[0][indication_col] or "").strip()
            if text and text.lower() != "nan":
                entry["indication"] = text
        manifest["studies"][str(study_id)] = entry
    return manifest


def _verify_results(
    harmonized_df: pd.DataFrame,
    results_dir: str,
    result_path: Callable[[pd.Series], str],
) -> None:
    """Fail early (with the missing relative paths) if the join won't resolve."""
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}. Pass results_dir= "
            "if rbclient.py wrote results somewhere else."
        )
    missing: list[str] = []
    for _, row in harmonized_df.iterrows():
        rel = result_path(row)
        if not os.path.isfile(os.path.join(results_dir, rel)):
            missing.append(rel)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)}/{len(harmonized_df)} expected result files are missing "
            f"under {results_dir} (e.g. {preview}). Check that the job finished and "
            "that the model/run names are right; override with results_dir= if needed."
        )


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
    view_col: str | None,
    indication_col: str | None,
    link: bool,
    check: bool,
    skip_submit: bool,
    write_manifest: bool,
    result_path: Callable[[pd.Series], str],
) -> str:
    """Stage → (manifest) → submit → watch, and return the results directory to score.

    When ``skip_submit`` is set the staging/submit/watch steps are skipped and the
    existing results directory is reused (re-scoring a finished run).
    """
    resolved_results = results_dir or client.output_dir(run_name, model)

    if not skip_submit:
        staging = staging_dir or os.path.join(client.data_dir, ".staging", run_name)
        manifest = _stage_studies(
            harmonized_df,
            base_image_dir,
            staging,
            image_col=image_col,
            study_col=study_col,
            view_col=view_col,
            indication_col=indication_col,
            link=link,
        )
        print(f"[reportbench] staged {len(manifest['studies'])} studies into {staging}")

        client.ensure_configured()
        if check:
            client.check()
        client.prepare(staging, run_name)
        # Make sure the service (possibly a different user) can create outputs under
        # the run folder — otherwise its output mkdir 403s / 500s.
        client.ensure_output_writable(run_name)
        if write_manifest:
            # Written into the prepared run folder (after prepare, which wipes+recreates it).
            manifest_path = os.path.join(client.run_dir(run_name), "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"[reportbench] wrote manifest for {len(manifest['studies'])} studies "
                  f"→ {manifest_path}")
        if getattr(client, "world_readable", False):
            # prepare copies with the subprocess umask (often o---); make the run
            # tree traversable/readable so the service (a different user) can read
            # it, else submit fails with an opaque HTTP 500. The output folder is
            # handled separately by ensure_output_writable (called after prepare).
            client.make_world_readable(run_name)
        client.submit(model, task, run_name)
        client.watch()
        client.results()

    _verify_results(harmonized_df, resolved_results, result_path)
    return resolved_results


def DatathonEmbeddingDataset(
    harmonized_df: pd.DataFrame,
    base_image_dir: str,
    *,
    client: ReportBenchClient,
    model: str,
    run_name: str = "run_embed",
    task: str = "embed",
    staging_dir: str | None = None,
    results_dir: str | None = None,
    image_col: str = "image_path",
    study_col: str = "study_id",
    view_col: str = "view_position",
    link: bool = True,
    check: bool = False,
    skip_submit: bool = False,
    emb_key: str = "global",
    emb_ext: str = ".npz",
    label_cols: Sequence[str] | None = None,
    cache_dir: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> EmbeddingResultsDataset:
    """Run the embed task through reportbench and return a ready dataset.

    Takes the same two inputs as an ordinary RadHarmony dataset — a harmonized
    frame (**one row per image/view**) and the raw image root — stages the
    studies, submits the *embed* task via :class:`ReportBenchClient`, waits, then
    returns an :class:`EmbeddingResultsDataset` pointed at the per-image
    ``<study>/<model>_<stem>.npz`` results (its ``global`` vector feeds the probe).

    The embed task submits **without a manifest** (views/indications don't affect
    embeddings), so studies are scanned positionally and every image is embedded.

    (This is a factory function, not a class: it does the submission work up front
    and hands back the underlying :class:`EmbeddingResultsDataset`.)

    Args:
        harmonized_df: One row per image, with ``study_col`` / ``image_col`` and
            the one-hot label columns (as produced by ``sample_data.ipynb``).
        base_image_dir: Root prepended to ``image_col`` to locate the raw images.
        client: Configured :class:`ReportBenchClient`.
        model: reportbench model id (e.g. ``"model-a"``). See ``client.models()``.
        run_name: Submission/run name; the run lands in ``<data_dir>/<run_name>``.
        task: reportbench task name (default ``"embed"``).
        staging_dir: Where to build the per-study submission layout (default
            ``<data_dir>/.staging/<run_name>``).
        results_dir: Override the results directory (default
            ``<data_dir>/<run_name>/_reportbench_out/<model>``).
        image_col, study_col, view_col: Column names in ``harmonized_df``.
        link: Symlink staged images (default) instead of copying. ``prepare`` still
            writes real copies into the run folder either way.
        check: Run ``rbclient.py check`` before submitting.
        skip_submit: Skip staging/submit/watch and reuse an existing
            ``results_dir`` (re-score a finished run).
        emb_key: Which array to read from each ``.npz`` (default ``"global"``).
        emb_ext: Embedding file extension (default ``".npz"``; ``".npy"`` for a
            bare-array layout).
        label_cols, cache_dir, dtype: Forwarded to :class:`EmbeddingResultsDataset`.
    """
    resolver = viewwise_embedding_path(model, ext=emb_ext)
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
        view_col=view_col,
        indication_col=None,
        link=link,
        check=check,
        skip_submit=skip_submit,
        write_manifest=False,
        result_path=resolver,
    )
    return EmbeddingResultsDataset(
        resolved,
        harmonized_df=harmonized_df,
        result_path=resolver,
        emb_key=emb_key,
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
    view_col: str = "view_position",
    indication_col: str | None = None,
    link: bool = True,
    check: bool = False,
    skip_submit: bool = False,
    cache_dir: str | None = None,
) -> ReportResultsDataset:
    """Run the report task through reportbench and return a ready dataset.

    The report-generation counterpart of :func:`DatathonEmbeddingDataset` (also a
    factory function, not a class): stages studies, submits the *report* task,
    waits, then returns a :class:`ReportResultsDataset` pointed at the per-study
    ``<study>/<model>_report.txt`` results (the frame's inline ``report`` column
    supplies the reference text). See that function for the shared arguments.

    A study-keyed ``manifest.json`` is written **only when** ``indication_col`` is
    given, so each study's indication is passed to the model as clinical context;
    without it the job is submitted with no manifest (views positional). The frame
    may have one row per image — it is de-duplicated to one row per study for
    reference scoring.
    """
    resolver = perstudy_report_path(model)
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
        view_col=view_col,
        indication_col=indication_col,
        link=link,
        check=check,
        skip_submit=skip_submit,
        write_manifest=indication_col is not None,
        result_path=resolver,
    )
    ref_df = harmonized_df.drop_duplicates(study_col).copy()
    return ReportResultsDataset(
        resolved,
        harmonized_df=ref_df,
        result_path=resolver,
        cache_dir=cache_dir,
    )


def DatathonPatchEmbeddingDataset(
    harmonized_df: pd.DataFrame,
    base_image_dir: str,
    *,
    client: ReportBenchClient,
    model: str,
    run_name: str = "run_embed",
    task: str = "embed",
    staging_dir: str | None = None,
    results_dir: str | None = None,
    image_col: str = "image_path",
    study_col: str = "study_id",
    view_col: str = "view_position",
    link: bool = True,
    check: bool = False,
    skip_submit: bool = False,
    emb_ext: str = ".npz",
    mask_col: str = "mask_path",
    mask_dir: str | None = None,
    grid_key: str = "grid",
    mask_size: int | None = None,
    feat_size: int | None = None,
    cache_dir: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> PatchSegResultsDataset:
    """Run the embed task through reportbench and return a SEGMENTATION dataset.

    The segmentation counterpart of :func:`DatathonEmbeddingDataset` (also a factory
    function, not a class). It submits the **same** *embed* task — one ``.npz`` per
    image holding ``global`` + ``patches`` (+ ``grid``) — but, instead of reading the
    pooled ``global`` vector, returns a :class:`PatchSegResultsDataset` that folds
    each image's ``patches`` into a dense ``[dim, H, W]`` feature map and serves it
    with the ground-truth ``mask_col`` mask. Classification and segmentation therefore
    share a single embed submission: run this with ``skip_submit=True`` +
    ``results_dir=`` (or the same ``run_name``) to reuse a run you already embedded
    for :func:`DatathonEmbeddingDataset`.

    Args:
        harmonized_df: One row per image, with ``study_col`` / ``image_col`` and a
            segmentation-mask column (``mask_col``).
        base_image_dir: Root prepended to ``image_col`` to locate the raw images.
        client, model, run_name, task, staging_dir, results_dir, image_col,
            study_col, view_col, link, check, skip_submit: as for
            :func:`DatathonEmbeddingDataset`.
        emb_ext: Embedding file extension (default ``".npz"``).
        mask_col: Column holding each image's ground-truth mask path.
        mask_dir: Root prepended to *relative* ``mask_col`` values (absolute paths
            are used as-is). Masks resolve against this rather than
            ``base_image_dir``/the embed outputs. ``None`` for absolute mask paths.
        grid_key: ``.npz`` key with the patch grid ``(H, W)`` (default ``"grid"``).
        mask_size: If set, masks are resized (nearest) to ``mask_size²``.
        feat_size: If set, feature maps are resized to ``feat_size²`` — needed when
            the patch grid varies per image.
        cache_dir, dtype: Forwarded to :class:`PatchSegResultsDataset`.
    """
    resolver = viewwise_embedding_path(model, ext=emb_ext)
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
        view_col=view_col,
        indication_col=None,
        link=link,
        check=check,
        skip_submit=skip_submit,
        write_manifest=False,
        result_path=resolver,
    )
    return PatchSegResultsDataset(
        resolved,
        harmonized_df=harmonized_df,
        result_path=resolver,
        mask_col=mask_col,
        mask_dir=mask_dir,
        grid_key=grid_key,
        mask_size=mask_size,
        feat_size=feat_size,
        cache_dir=cache_dir,
        dtype=dtype,
    )
