"""Visualization helpers for radharmony_hands_on_evaluation.ipynb.

Each function pairs images with what an evaluator produced — classification
probabilities, segmentation masks, VQA answers, generated reports — so the
notebook cells stay one-liners. Kept out of the notebook to keep the narrative
readable; import and call from there.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import GroupKFold

from radharmony.evaluator.language._report_parser import parse_report_section


def show_predictions(probe, probe_ds, montgomery_dir, *, label="tb", n_splits=5):
    """Classification: read P(label) off the probe for a few held-out images.

    Reuses the embeddings ``probe.evaluate()`` already cached, does one honest
    ``GroupKFold`` split (fit the head on the training patients, predict on the
    held-out fold), and shows two positive + two negative validation cases.
    """
    feats, labels = probe._load_or_extract(probe_ds)
    train_idx, val_idx = next(
        GroupKFold(n_splits=n_splits).split(feats, groups=probe._patient_ids)
    )
    probe._fit_final_models(feats[train_idx], labels[train_idx])
    clf = probe.final_models_[label]

    val_labels = labels[val_idx, 0]
    show = np.concatenate([val_idx[val_labels == 1][:2], val_idx[val_labels == 0][:2]])
    probs = clf.predict_proba(feats[show])[:, 1]

    # Image paths line up row-for-row with feats (extract_embeddings order).
    df = probe_ds._get_harmonized_df().dropna(subset=["image_path"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 5))
    for ax, i, p in zip(np.atleast_1d(axes), show, probs):
        img_path = os.path.join(montgomery_dir, df.loc[i, "image_path"])
        ax.imshow(Image.open(img_path).convert("L"), cmap="gray")
        ax.axis("off")
        truth = label.upper() if labels[i, 0] == 1 else "normal"
        pred = label.upper() if p >= 0.5 else "normal"
        ax.set_title(f"P({label}) = {p:.2f}\npred: {pred}\ntrue: {truth}", fontsize=9)
    plt.tight_layout()
    plt.show()


def show_masks(seg_probe, seg_ds, *, n=4):
    """Segmentation: image, ground-truth mask, predicted mask for a few images.

    Trains a single deployment head (skipping if ``evaluate()`` already stored
    one) and predicts on the honest inner-validation slice ``fit()`` held out.
    """
    if getattr(seg_probe, "final_head_", None) is None:
        seg_probe.fit(seg_ds)
    val_idx = seg_probe.inner_val_indices(seg_ds)[:n]
    panels = seg_probe.predict(seg_ds, indices=val_idx, return_images=True)

    fig, axes = plt.subplots(len(panels), 3, figsize=(9, 3 * len(panels)))
    axes = np.atleast_2d(axes)
    for row, p in enumerate(panels):
        cols = [
            ("image", None),
            ("ground-truth lung", p["y_true"]),
            ("predicted lung", p["y_pred"]),
        ]
        for col, (title, mask) in enumerate(cols):
            ax = axes[row, col]
            ax.imshow(p["image"], cmap="gray")
            if mask is not None:
                ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.5)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_answers(vqa_ds, answerer, *, n=4):
    """VQA: pair a few images with the reference answer and the model's answer."""
    td = vqa_ds.get_datasets(n_splits=None)
    samples = [td[i] for i in range(n)]
    preds = answerer([s["img"] for s in samples], [s["question"] for s in samples])

    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 5))
    for ax, s, pred in zip(np.atleast_1d(axes), samples, preds):
        ax.imshow(Image.open(s["img"]).convert("L"), cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"Q: {s['question']}\nref: {s['answer']}\npred: {pred}",
            fontsize=8,
            wrap=True,
        )
    plt.tight_layout()
    plt.show()


def show_reports(gen_ds, generator, *, n=3, section="findings"):
    """Report generation: image beside its reference vs generated findings."""
    td = gen_ds.get_datasets(n_splits=None)
    samples = [td[i] for i in range(n)]
    gen_reports = generator([s["img"] for s in samples], [None] * len(samples))

    fig, axes = plt.subplots(
        len(samples),
        2,
        figsize=(11, 4.2 * len(samples)),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    axes = np.atleast_2d(axes)
    for row, (s, hyp) in enumerate(zip(samples, gen_reports)):
        axes[row, 0].imshow(Image.open(s["img"]).convert("L"), cmap="gray")
        axes[row, 0].axis("off")
        ref = parse_report_section(str(s["report"]), section)
        axes[row, 1].axis("off")
        axes[row, 1].text(
            0,
            1,
            f"REFERENCE {section}:\n{ref}\n\nGENERATED {section}:\n{hyp}",
            va="top",
            ha="left",
            fontsize=8,
            wrap=True,
            transform=axes[row, 1].transAxes,
        )
    plt.tight_layout()
    plt.show()
