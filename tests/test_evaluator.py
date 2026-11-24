"""Evaluator tests on toy data (CPU, no NAS).

Each test names the finding that it guards (see ``reports/eval/index.json``).
The expected values come from the data that the test builds, or from the
decision in ``docs/wiki/design.md``.
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd
import pytest
import torch

import radharmony.evaluator as rev
from radharmony.evaluator import LinearProbeEvaluator

from _toy_eval import CPU_KW, toy_real_cls_ds


def test_registry_docstring_lists_real_evaluators():
    """V-A-2, R-code-13: an output list in the registry docstring matches the registry."""
    doc = ast.get_docstring(ast.parse(open(rev._registry.__file__).read()))
    for listed in re.findall(r"list_evaluators\(\)\)\s*#\s*(\[.*?\])", doc):
        assert ast.literal_eval(listed) == rev.list_evaluators()


def test_num_workers_zero_runs():
    """V-E-5: num_workers=0 is a normal DataLoader value. The evaluator must run with it."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    y = (x[:, :1] > 0).astype(float)
    ds = toy_real_cls_ds(x, y, ["a"])
    df = LinearProbeEvaluator(torch.nn.Identity(), dataset=ds, n_folds=2, **CPU_KW).evaluate()
    assert set(df["fold"]) == {0, 1}


def test_f1_is_zero_with_no_true_positive():
    """V-E-10: no true positive gives F1 = 0 (sklearn f1_score), not NaN."""
    from sklearn.metrics import f1_score

    from radharmony.evaluator.metrics.classification import compute_metrics

    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.1])  # with 0.5, only the 2 negatives are positive
    m = compute_metrics(y, s, "fixed:0.5")
    assert m["f1"] == f1_score(y, (s >= 0.5).astype(int)) == 0.0


def test_youden_threshold_is_finite():
    """V-E-11: the Youden threshold skips the +inf entry of roc_curve."""
    from radharmony.evaluator.metrics.classification import compute_metrics

    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])  # the score runs against the label
    m = compute_metrics(y, s, "youden")
    assert np.isfinite(m["threshold"])
    # Every finite cut has TPR - FPR <= 0. The lowest score gives TPR = FPR = 1.
    assert m["threshold"] == 0.1 and m["tpr"] == 1.0


def _separable(n, rng, bad_frac=0.0, bad_value=-1.0):
    """Label 'a' is separable on feature 0 (true AUROC 1.0). ``bad_frac`` of the rows get ``bad_value``."""
    y = rng.integers(0, 2, n).astype(np.float32)
    x = (rng.normal(size=(n, 4)) * 0.3).astype(np.float32)
    x[:, 0] += 2 * y - 1
    x[:, 1] += 3.0
    y[rng.random(n) < bad_frac] = bad_value
    return x, y[:, None]


def test_minus_one_train_label_is_not_a_class():
    """V-E-1: a label is 0, 1 or NaN (D-05). A -1 train label (CheXpert "uncertain") is dropped."""
    from radharmony.evaluator import PrototypeProbeEvaluator, SVMProbeEvaluator

    from _toy_eval import ToyClsDS

    rng = np.random.default_rng(0)
    train, test = ToyClsDS(*_separable(300, rng, 0.3), ["a"]), ToyClsDS(*_separable(150, rng), ["a"])
    for cls in (LinearProbeEvaluator, SVMProbeEvaluator, PrototypeProbeEvaluator):
        df = cls(torch.nn.Identity(), train_dataset=train, test_dataset=test, **CPU_KW).evaluate()
        assert df.loc[(df["label"] == "a") & (df["bootstrap"] == -1), "auroc"].iloc[0] == 1.0, cls.__name__


def _by_feature(n, cols, rng):
    """Label 'a' comes from feature 0, 'b' from feature 1, 'c' from feature 2. Columns follow ``cols``."""
    from _toy_eval import ToyClsDS

    x = rng.normal(size=(n, 6)).astype(np.float32)
    x[:, 3] += 3.0  # keeps the feature norms away from 0
    src = {"a": 0, "b": 1, "c": 2}
    y = np.stack([(x[:, src[c]] > 0).astype(np.float32) for c in cols], 1)
    return ToyClsDS(x, y, cols)


_FIXED_SPLIT_PROBES = ["LinearProbeEvaluator", "SVMProbeEvaluator", "KNNProbeEvaluator",
                       "PrototypeProbeEvaluator", "FinetuneEvaluator"]


def _probe(name, **kw):
    cls = getattr(rev, name)
    if name == "FinetuneEvaluator":
        kw = {"freeze_backbone": True, "epochs": 20, "lr": 1e-2, "val_fraction": 0.0, **kw}
    return cls(torch.nn.Identity(), **kw, **CPU_KW)


@pytest.mark.parametrize("name", _FIXED_SPLIT_PROBES)
def test_threshold_comes_from_validation_data(name):
    """V-E-3, R-code-2 (D-29): the test labels do not pick the threshold.

    Two test sets with the same images but flipped labels give the same
    thresholds. Every bootstrap sample uses the threshold of its point estimate.
    """
    from _toy_eval import ToyClsDS

    rng = np.random.default_rng(0)
    train, test = _by_feature(300, ["a", "b"], rng), _by_feature(200, ["a", "b"], rng)
    flipped = ToyClsDS(test.feats, 1 - test.labels, test.LABEL_COLS)
    thresholds = []
    for t in (test, flipped):
        df = _probe(name, train_dataset=train, test_dataset=t, n_bootstrap=5, val_fraction=0.1).evaluate()
        df = df[df["label"] != "macro_average"]
        point = df[df["bootstrap"] == -1].set_index("label")["threshold"]
        assert np.isfinite(point).all()
        for _, sample in df.groupby("bootstrap"):
            np.testing.assert_allclose(sample.set_index("label")["threshold"], point)
        thresholds.append(point)
    np.testing.assert_allclose(thresholds[0], thresholds[1])


def test_bootstrap_draws_patients():
    """D-32: a bootstrap sample keeps all images of each drawn patient."""
    from radharmony.evaluator.sampling import bootstrap_indices

    # Patient p0 has 20 follow-up images, 30 patients have 1. The rows are mixed.
    groups = np.array(["p0"] * 20 + [f"q{i}" for i in range(30)])
    groups = groups[np.random.default_rng(1).permutation(len(groups))]
    p0_rows = set(np.flatnonzero(groups == "p0"))
    for idx in bootstrap_indices(len(groups), 50, seed=3, groups=groups):
        n_p0 = sum(int(i in p0_rows) for i in idx)
        assert n_p0 % 20 == 0  # the 20 images go in or out together
        assert n_p0 // 20 + (len(idx) - n_p0) == 31  # 31 patients drawn
    # One row per patient: the same draws as a draw of rows (D-32 example).
    ones = np.array([f"r{i}" for i in range(40)])
    by_patient = list(bootstrap_indices(40, 5, 7, groups=ones))
    by_row = list(bootstrap_indices(40, 5, 7))
    assert all(np.array_equal(a, b) for a, b in zip(by_patient, by_row, strict=True))


class _ToyReportDS:
    """Report dataset: each sample has an image and its report text."""

    def __init__(self, n, patient_ids):
        self.df = pd.DataFrame({"image_path": [f"img_{i}.png" for i in range(n)], "patient_id": patient_ids})

    def _get_harmonized_df(self):
        return self.df

    def get_datasets(self, n_splits=None, num_cores=2):
        return [{"img": torch.zeros(3), "report": f"FINDINGS: finding {i}."} for i in range(len(self.df))]


@pytest.mark.parametrize("name", _FIXED_SPLIT_PROBES + ["ZeroShotEvaluator", "LinearProbeSegEvaluator",
                                                        "ReportGenerationEvaluator"])
def test_bootstrap_uses_test_patients(name, monkeypatch):
    """D-32: every evaluator draws the test patients in its bootstrap."""
    import importlib

    import radharmony.evaluator.sampling as sampling
    from _toy_eval import ToyClsDS, ToySegDS, toy_seg_data

    seen = []

    def spy(n_rows, n_bootstrap, seed, groups=None):
        seen.append(None if groups is None else [str(g) for g in groups])
        return sampling.bootstrap_indices(n_rows, n_bootstrap, seed, groups)

    for mod in ("classification._probe_utils", "classification.zero_shot",
                "segmentation.linear_probe", "language.report_generation"):
        monkeypatch.setattr(importlib.import_module(f"radharmony.evaluator.{mod}"), "bootstrap_indices",
                            spy, raising=False)

    test_pids = [f"t{i // 2}" for i in range(40)]  # 2 images per test patient
    rng = np.random.default_rng(0)
    if name == "ZeroShotEvaluator":
        t = _by_feature(40, ["a"], rng)
        ev = rev.ZeroShotEvaluator(torch.nn.Identity(), lambda texts: torch.ones(len(texts), 6), {"a": ["a"]},
                                   test_dataset=ToyClsDS(t.feats, t.labels, ["a"], test_pids), n_bootstrap=2,
                                   **CPU_KW)
    elif name == "LinearProbeSegEvaluator":
        ev = rev.LinearProbeSegEvaluator(torch.nn.Identity(), train_dataset=ToySegDS(*toy_seg_data(n=16)),
                                         test_dataset=ToySegDS(*toy_seg_data(n=40, seed=1), test_pids),
                                         epochs=1, batch_size=4, n_bootstrap=2, **CPU_KW)
    elif name == "ReportGenerationEvaluator":
        rg = importlib.import_module("radharmony.evaluator.language.report_generation")
        monkeypatch.setattr(rg, "compute_report_metrics", lambda refs, hyps, metrics: {"n": len(refs)})
        ev = rev.ReportGenerationEvaluator(lambda imgs: ["text"] * len(imgs), dataset=_ToyReportDS(40, test_pids),
                                           n_bootstrap=2, device="cpu", num_workers=0, metrics=("n",))
    else:
        t = _by_feature(40, ["a"], rng)
        ev = _probe(name, train_dataset=_by_feature(100, ["a"], rng), n_bootstrap=2,
                    test_dataset=ToyClsDS(t.feats, t.labels, ["a"], test_pids))
    ev.evaluate()
    assert seen and all(g == test_pids for g in seen)


def _zero_shot(**kw):
    rng = np.random.default_rng(0)
    return rev.ZeroShotEvaluator(torch.nn.Identity(), lambda texts: torch.eye(6)[: len(texts)],
                                 {"a": ["a"]}, test_dataset=_by_feature(40, ["a"], rng), **kw, **CPU_KW)


def test_zero_shot_bootstrap_has_no_wrong_warning():
    """V-E-14 (D-19): zero-shot with test_dataset= does not say that n_bootstrap is ignored."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev = _zero_shot(n_bootstrap=3)
    assert not [w for w in caught if "ignored" in str(w.message)]
    df = ev.evaluate()
    assert sorted(set(df["bootstrap"])) == [-1, 0, 1, 2]


_THRESHOLD_METRICS = ["f1", "accuracy", "balanced_accuracy", "tpr", "tnr", "ppv", "npv", "mcc", "threshold"]


def test_zero_shot_threshold_metrics_are_nan_without_a_fixed_threshold():
    """R-code-2 (D-29): zero-shot has no validation data, so the test labels do not pick a threshold."""
    with pytest.warns(UserWarning, match="threshold metrics are NaN"):
        ev = _zero_shot(n_bootstrap=2)
    df = ev.evaluate()
    df = df[df["label"] == "a"]
    assert df[_THRESHOLD_METRICS].isna().all().all()
    assert df["auroc"].notna().all()


def test_zero_shot_uses_the_fixed_threshold():
    """R-code-2 (D-29): threshold_strategy='fixed:<value>' sets the threshold of every row."""
    from sklearn.metrics import f1_score

    df = _zero_shot(n_bootstrap=2, threshold_strategy="fixed:0").evaluate()
    df = df[df["label"] == "a"]
    assert (df["threshold"] == 0.0).all()
    t = _by_feature(40, ["a"], np.random.default_rng(0))
    score = t.feats[:, 0] / np.linalg.norm(t.feats, axis=1)  # cosine with the prompt e0
    point = df[df["bootstrap"] == -1].iloc[0]
    assert point["f1"] == pytest.approx(f1_score(t.labels[:, 0], (score >= 0).astype(int)))


def test_zero_shot_warns_with_negative_prompts_to_use_fixed_0():
    """D-29: with negative prompts, the warning names threshold_strategy='fixed:0'."""
    with pytest.warns(UserWarning, match="fixed:0"):
        _zero_shot(negative_prompts={"a": ["no a"]})


def test_zero_shot_warns_for_l2_normalize_false():
    """D-17: zero-shot always uses cosine similarity, so l2_normalize=False has no effect. It warns."""
    import warnings

    with pytest.warns(UserWarning, match="l2_normalize"):
        _zero_shot(l2_normalize=False, threshold_strategy="fixed:0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _zero_shot(threshold_strategy="fixed:0")
    assert not [w for w in caught if "l2_normalize" in str(w.message)]


def test_zero_shot_prompt_lookup_ignores_case():
    """V-E-22: labels= ignores case, so the prompt lookup ignores case too."""
    from _toy_eval import ToyClsDS

    t = _by_feature(40, ["a"], np.random.default_rng(0))
    ds = ToyClsDS(t.feats, t.labels, ["Pneumonia"])
    ev = rev.ZeroShotEvaluator(torch.nn.Identity(), lambda texts: torch.eye(6)[: len(texts)],
                               {"pneumonia": ["pneumonia"]}, negative_prompts={"PNEUMONIA": ["normal"]},
                               labels=["pneumonia"], test_dataset=ds, **CPU_KW)
    df = ev.evaluate()
    assert "Pneumonia" in set(df["label"])


def test_finetune_ce_scores_with_softmax():
    """V-E-19: loss='ce' trains a softmax model, so the scores are softmax (each row sums to 1)."""
    from _toy_eval import ToyClsDS

    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 6)).astype(np.float32)
    y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, 60)]  # one class per row
    ds = ToyClsDS(x, y, ["c0", "c1", "c2"])
    ev = rev.FinetuneEvaluator(torch.nn.Identity(), dataset=ds, loss="ce", freeze_backbone=True,
                               epochs=2, val_fraction=0.0, **CPU_KW)
    ev.fit()
    prob = np.stack([p["prob"] for p in ev.predict(ds)])
    np.testing.assert_allclose(prob.sum(axis=1), 1.0, rtol=1e-5)


def test_finetune_predict_takes_the_trained_label_columns_by_name():
    """V-E-30: predict() takes the dataset columns named in final_labels_, and raises when one is missing."""
    from _toy_eval import ToyClsDS

    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 4)).astype(np.float32)
    y = (rng.random((60, 3)) > 0.5).astype(np.float32)
    ev = rev.FinetuneEvaluator(torch.nn.Identity(), dataset=ToyClsDS(x, y[:, :2], ["a", "b"]), epochs=1,
                               freeze_backbone=True, **CPU_KW)
    ev.fit()
    assert ev.final_labels_ == ["a", "b"]
    out = ev.predict(ToyClsDS(x, y, ["a", "b", "c"]))  # one more label: keep "a" and "b"
    np.testing.assert_array_equal(np.stack([r["y_true"] for r in out]), y[:, :2])
    with pytest.raises(ValueError, match="'b'"):
        ev.predict(ToyClsDS(x, y[:, [0, 2]], ["a", "c"]))  # same count, but "b" is missing


def test_finetune_ce_with_one_label_column_raises():
    """V-E-27 (D-47): a softmax over 1 logit is always 1, so loss='ce' with 1 label column cannot train."""
    from _toy_eval import ToyClsDS

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6)).astype(np.float32)
    ds = ToyClsDS(x, (x[:, :1] > 0).astype(np.float32), ["tuberculosis"])
    ev = rev.FinetuneEvaluator(torch.nn.Identity(), dataset=ds, loss="ce", freeze_backbone=True,
                               epochs=1, val_fraction=0.0, **CPU_KW)
    with pytest.raises(ValueError, match="bce"):
        ev.fit()


def _norm_data():
    """Features with very different norms. Label 'a' comes from feature 0."""
    from _toy_eval import ToyClsDS

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 6)).astype(np.float32) * rng.uniform(0.2, 8.0, size=(80, 1)).astype(np.float32)
    x[:, 1] += 2.0
    return ToyClsDS(x, (x[:, :1] > 0).astype(np.float32), ["a"])


@pytest.mark.parametrize("l2", [None, True, False])
def test_finetune_frozen_head_sees_l2_normalized_features(l2):
    """V-E-20, R-code-4 (D-17): a frozen-backbone fine-tune normalizes by default and obeys l2_normalize."""
    ds = _norm_data()
    head = torch.nn.Linear(6, 1)
    norms = []
    head.register_forward_pre_hook(lambda m, args: norms.append(args[0].norm(dim=1)))
    kw = {} if l2 is None else {"l2_normalize": l2}
    ev = rev.FinetuneEvaluator(torch.nn.Identity(), dataset=ds, n_folds=2, freeze_backbone=True, head=head,
                               epochs=1, val_fraction=0.0, **kw, **CPU_KW)
    ev.evaluate()
    unit = torch.allclose(torch.cat(norms), torch.ones(1), atol=1e-4)
    assert unit == (l2 is not False)


@pytest.mark.parametrize("l2", [True, False])
def test_prototype_obeys_l2_normalize(l2):
    """V-E-20 (D-17): l2_normalize=False makes the prototype centroids from raw features."""
    ds = _norm_data()
    ev = rev.PrototypeProbeEvaluator(torch.nn.Identity(), dataset=ds, n_folds=2, l2_normalize=l2,
                                     store_final_model=True, **CPU_KW)
    ev.evaluate()
    x, y = ds.feats, ds.labels[:, 0]
    if l2:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    c_pos = x[y == 1].mean(axis=0)
    np.testing.assert_allclose(ev.final_models_["a"][0], c_pos / np.linalg.norm(c_pos), rtol=1e-5)


@pytest.mark.parametrize("name", ["FinetuneEvaluator", "LinearProbeSegEvaluator"])
def test_epoch_with_no_val_signal_does_not_stop_training(name):
    """V-E-16: an epoch with no validation signal (-inf, for example no positive) does not count for patience."""
    from _toy_eval import ToySegDS, toy_seg_data

    calls = []
    base = getattr(rev, name)

    class NoSignal(base):
        def _val_metric(self, *args, **kwargs):
            calls.append(1)
            return (-float("inf"), float("nan")) if name == "FinetuneEvaluator" else -float("inf")

    if name == "FinetuneEvaluator":
        ev = NoSignal(torch.nn.Identity(), dataset=_by_feature(40, ["a"], np.random.default_rng(0)),
                      freeze_backbone=True, epochs=8, early_stop_patience=2, **CPU_KW)
    else:
        ev = NoSignal(torch.nn.Identity(), dataset=ToySegDS(*toy_seg_data(n=16)), epochs=8,
                      early_stop_patience=2, batch_size=4, **CPU_KW)
    ev.fit()
    assert len(calls) == 8  # one validation per epoch, all epochs run


@pytest.mark.parametrize("name", ["LinearProbeSegEvaluator", "ConvProbeSegEvaluator"])
def test_seg_head_trains_when_train_part_is_smaller_than_batch(name):
    """V-E-17: a train part smaller than batch_size still trains the head."""
    from _toy_eval import ToySegDS, toy_seg_data

    changed = []

    class Spy(getattr(rev, name)):
        def _train_one(self, head, train_loader, val_loader=None):
            before = [p.detach().clone() for p in head.parameters()]
            out = super()._train_one(head, train_loader, val_loader)
            changed.append(any(not torch.equal(a, b) for a, b in zip(before, head.parameters(), strict=True)))
            return out

    ev = Spy(torch.nn.Identity(), train_dataset=ToySegDS(*toy_seg_data(n=16)),
             test_dataset=ToySegDS(*toy_seg_data(n=8, seed=1)), n_train_samples=[4], batch_size=8,
             epochs=2, val_fraction=0.0, **CPU_KW)
    ev.evaluate()
    assert changed == [True]


def test_seg_index_mask_targets_do_not_depend_on_the_batch():
    """V-E-6: an index mask {0..K-1} gives the same targets in every batch."""
    ev = rev.LinearProbeSegEvaluator(torch.nn.Identity(), dataset=_FakeAny(), num_classes=3, **CPU_KW)
    mask = torch.tensor([[[0, 1, 1, 0]]], dtype=torch.float32)  # [B=1, H=1, W=4], no class 2
    with_2 = torch.tensor([[[0, 2, 1, 0]]], dtype=torch.float32)
    assert ev._mask_to_target(mask).tolist() == [[[0, 1, 1, 0]]]
    assert ev._mask_to_target(torch.cat([mask, with_2])).tolist() == [[[0, 1, 1, 0]], [[0, 2, 1, 0]]]
    with pytest.raises(ValueError, match="class indices"):
        ev._mask_to_target(mask * 255)  # gray levels for 3 classes: the mapping is not known
    binary = rev.LinearProbeSegEvaluator(torch.nn.Identity(), dataset=_FakeAny(), num_classes=2, **CPU_KW)
    assert binary._mask_to_target(mask * 255).tolist() == [[[0, 1, 1, 0]]]  # {0, 255} binary PNG
    # V-E-32 (D-14): in a binary mask, any value above 0 is foreground, also a 1 next to a 2.
    assert binary._mask_to_target(torch.tensor([[[0, 1, 2, 2]]], dtype=torch.float32)).tolist() == [[[0, 1, 1, 1]]]


def test_seg_ids_use_the_dataset_rows():
    """V-E-9 (D-33): rows with an empty mask_path are not in the dataset, so they are not in the ids."""
    from _toy_eval import toy_real_seg_ds, toy_seg_data

    imgs, masks = toy_seg_data(n=20)
    ds = toy_real_seg_ds(imgs, masks, patient_ids=[f"p{i // 2}" for i in range(20)], null_mask_rows=(0, 1))
    ev = rev.LinearProbeSegEvaluator(torch.nn.Identity(), dataset=ds, n_folds=2, epochs=1, batch_size=4,
                                     **CPU_KW)
    df = ev.evaluate()  # raised "inconsistent numbers of samples: [18, 20]"
    assert set(df["fold"]) == {0, 1}
    ev.fit()
    assert ev.predict(ds, indices=[0])[0]["image_id"] == "img_2"  # the first row with a mask


def test_prediction_ids_and_pngs_are_one_per_image(tmp_path):
    """V-E-24: CheXpert-style paths share the file name. Each image keeps its own id and PNG."""
    from _toy_eval import ToySegDS, toy_seg_data

    ds = ToySegDS(*toy_seg_data(n=12))
    ds.df["image_path"] = [f"patient{i}/study1/view1_frontal.jpg" for i in range(12)]
    expected = sorted(f"patient{i}/study1/view1_frontal" for i in range(12))
    assert sorted(rev.FinetuneEvaluator._image_ids(ds)) == expected
    ev = rev.LinearProbeSegEvaluator(torch.nn.Identity(), dataset=ds, n_folds=2, epochs=1, batch_size=4,
                                     save_predictions=True, output_dir=str(tmp_path), **CPU_KW)
    ev.evaluate()
    assert len(list((tmp_path / "GT").rglob("*.png"))) == 12
    ev.fit()
    assert sorted(r["image_id"] for r in ev.predict(ds)) == expected


def test_image_ids_cut_only_an_image_file_ending():
    """V-E-24 (D-46): a 3-D study folder named by a dotted UID keeps all of its name."""
    from _toy_eval import ToySegDS, toy_seg_data
    from radharmony.evaluator.sampling import image_ids

    ds = ToySegDS(*toy_seg_data(n=4))
    ds.df["image_path"] = ["1.2.826.0.1.3680043.6200", "1.2.826.0.1.3680043.27262", "p1/s1/a.nii.gz", "p2/b.dcm"]
    assert image_ids(ds) == ["1.2.826.0.1.3680043.6200", "1.2.826.0.1.3680043.27262", "p1/s1/a", "p2/b"]


@pytest.mark.parametrize("report, section, expected", [
    # V-E-23: a lowercase "findings:" or "impression:" inside a sentence is not a header.
    ("HISTORY: r/o new findings: none reported.\nFINDINGS: Clear.\nIMPRESSION: Normal.", "findings", "Clear."),
    ("INDICATION: Evaluate for findings: cough.\n\nFINDINGS: Lungs clear.\n\nIMPRESSION: Normal.",
     "findings", "Lungs clear."),
    ("FINDINGS: Compared to prior, see impression: unchanged opacity.\n\nIMPRESSION: Stable.",
     "findings", "Compared to prior, see impression: unchanged opacity."),
    ("FINDINGS: Compared to prior, see impression: unchanged opacity.\n\nIMPRESSION: Stable.",
     "impression", "Stable."),
    ("IMPRESSION: Small effusion. Findings: stable.", "both", "Small effusion. Findings: stable."),
    # V-E-8: the findings stop at the next section header, also on the same line.
    ("FINDINGS: Lungs clear.\n\nCOMPARISON: None.\n\nIMPRESSION: Normal.", "findings", "Lungs clear."),
    ("FINDINGS: Lungs clear.\n\nTECHNIQUE: PA and lateral.\n\nIMPRESSION: Normal.", "findings", "Lungs clear."),
    ("FINDINGS: Lungs clear. Heart normal. IMPRESSION: No acute disease.", "findings",
     "Lungs clear. Heart normal."),
    ("FINDINGS: Lungs clear. Heart normal. IMPRESSION: No acute disease.", "impression", "No acute disease."),
    ("EXAMINATION: CXR\n\nINDICATION: cough\n\nFINDINGS: Lungs clear. Heart normal.\n\n"
     "IMPRESSION: No acute disease.\n", "both", "Lungs clear. Heart normal. No acute disease."),
    # Q-9 (D-38): the impression stops at a FINDINGS header after it, so "both" has each text once.
    ("IMPRESSION: Stable. FINDINGS: Heart is normal.", "impression", "Stable."),
    ("IMPRESSION: Stable.\n\nFINDINGS: Heart is normal.", "both", "Heart is normal. Stable."),
    # V-E-28 (D-38): with neither section, the fallback keeps the whole report. A lowercase word in a line
    # is not a header, so the fallback does not cut there.
    ("INDICATION: cough.\nChest x-ray compared with prior. No new findings: lungs are clear.", "findings",
     "INDICATION: cough. Chest x-ray compared with prior. No new findings: lungs are clear."),
    ("EXAMINATION: CHEST PA\nHeart size normal. Overall impression: no acute process.", "impression",
     "EXAMINATION: CHEST PA Heart size normal. Overall impression: no acute process."),
])
def test_report_parser_headers(report, section, expected):
    """V-E-23, V-E-8: a header starts a line, or is in capitals. A section stops at the next header."""
    from radharmony.evaluator.language._report_parser import parse_report_section

    assert parse_report_section(report, section) == expected


def test_seg_dice_rule_on_empty_ground_truth():
    """V-E-12: the written rule for an image with no foreground in the ground truth.

    Empty prediction: NaN, so the image does not count in the mean. Any
    foreground pixel in the prediction: 0. A non-empty ground truth gives
    2TP / (|pred| + |gt|), as MONAI does.
    """
    from radharmony.evaluator.metrics.segmentation import compute_metrics as seg_metrics

    empty, one_px = np.zeros((4, 4), int), np.zeros((4, 4), int)
    one_px[0, 0] = 1
    assert np.isnan(seg_metrics(empty, empty, 2)[1]["dice"])
    assert seg_metrics(empty, one_px, 2)[1]["dice"] == 0.0
    gt = np.zeros((4, 4), int)
    gt[0, :2] = 1
    assert seg_metrics(gt, one_px, 2)[1]["dice"] == 2 * 1 / (1 + 2)
    doc = __import__("radharmony.evaluator.metrics.segmentation", fromlist=["x"]).__doc__
    assert "empty" in doc and "NaN" in doc


@pytest.mark.parametrize("page", ["linear_probe", "svm_probe", "knn_probe", "prototype_probe"])
def test_store_final_model_docs_match_the_code(page, tmp_path):
    """V-E-13: the frozen probes keep the final models in memory. The wiki row must not promise a file."""
    import pathlib

    name = {"linear_probe": "LinearProbeEvaluator", "svm_probe": "SVMProbeEvaluator",
            "knn_probe": "KNNProbeEvaluator", "prototype_probe": "PrototypeProbeEvaluator"}[page]
    rng = np.random.default_rng(0)
    ev = _probe(name, train_dataset=_by_feature(60, ["a"], rng), test_dataset=_by_feature(30, ["a"], rng),
                store_final_model=True, output_dir=str(tmp_path))
    ev.evaluate()
    assert list(tmp_path.iterdir()) == [] and set(ev.final_models_) == {"a"}
    wiki = pathlib.Path(__file__).resolve().parents[1] / "docs" / "wiki" / "reference" / "evaluators" / f"{page}.md"
    row = next(ln for ln in wiki.read_text().splitlines() if ln.startswith("| `store_final_model`"))
    assert "final_models_" in row and "Save" not in row


def test_finetune_docstring_does_not_promise_cached_features():
    """V-E-13: with freeze_backbone=True the encoder runs on every batch. No feature cache exists."""
    assert "cached embeddings" not in rev.classification.finetune.__doc__


# ---------------------------------------------------------------------------
# Base-code fixes of 2026-10-29 (approved by Frank Li).
# ---------------------------------------------------------------------------
def test_embedding_cache_is_checked(tmp_path):
    """V-E-2: a cache works for another label list. Other images or another encoder raise.

    Decided by Frank Li on 2026-10-29: raise, do not build the cache again.
    """
    rng = np.random.default_rng(0)
    cache = str(tmp_path / "emb")
    ds = _by_feature(60, ["a", "b", "c"], rng)
    _probe("LinearProbeEvaluator", dataset=ds, n_folds=2, embedding_cache=cache).evaluate()
    df = _probe("LinearProbeEvaluator", dataset=ds, n_folds=2, labels=["c"], embedding_cache=cache).evaluate()
    assert set(df["label"]) == {"c", "macro_average"}
    other_rows = _by_feature(30, ["a", "b", "c"], rng)
    with pytest.raises(ValueError, match="other images"):
        _probe("LinearProbeEvaluator", dataset=other_rows, n_folds=2, embedding_cache=cache).evaluate()
    ev = _probe("LinearProbeEvaluator", dataset=ds, n_folds=2, embedding_cache=cache)
    ev.image_encoder = torch.nn.Sequential(torch.nn.Identity())
    with pytest.raises(ValueError, match="comes from"):
        ev.evaluate()


def test_embedding_cache_rows_follow_the_samples(tmp_path):
    """V-E-2, V-E-9: with output_mask=True, a row with no mask gives no sample.

    The cache and the patient ids use the same rows as the samples.
    """
    ds = _by_feature(3, ["a"], np.random.default_rng(0))
    ds.df["patient_id"] = ["p1", "p2", "p3"]
    ds.df["mask_path"] = ["m1.png", None, "m3.png"]
    ds._cols = {"cls": ["a"], "mask": "mask_path"}
    keep = ds.df["mask_path"].notna().to_numpy()
    ds.feats, ds.labels = ds.feats[keep], ds.labels[keep]
    for _ in range(2):  # extract, then load
        ev = _probe("LinearProbeEvaluator", dataset=ds, embedding_cache=str(tmp_path / "emb"))
        feats, _ = ev._load_or_extract(ds)
        assert len(feats) == 2
        assert list(ev._patient_ids) == ["p1", "p3"]


def test_embedding_cache_tells_wrapped_encoders_apart(tmp_path):
    """V-E-2: every backbone recipe is an ImageEncoderWrapper. The check uses the model inside it."""
    from radharmony.evaluator.wrappers import ImageEncoderWrapper

    ds = _by_feature(20, ["a"], np.random.default_rng(0))

    def run(seed):
        torch.manual_seed(seed)
        encoder = ImageEncoderWrapper(torch.nn.Linear(6, 6))
        return LinearProbeEvaluator(encoder, dataset=ds, embedding_cache=str(tmp_path / "emb"), **CPU_KW)

    run(0)._load_or_extract(ds)
    run(0)._load_or_extract(ds)  # the same weights read the cache
    with pytest.raises(ValueError, match="comes from Linear:"):
        run(1)._load_or_extract(ds)


def test_an_old_embedding_cache_raises(tmp_path):
    """V-E-2: a cache file from before 2026-10-29 has no image or encoder check. So it raises."""
    import pickle

    ds = _by_feature(10, ["a"], np.random.default_rng(0))
    with open(tmp_path / "emb.pkl", "wb") as f:
        pickle.dump({"feats": ds.feats, "labels": ds.labels, "labels_names": ["a"]}, f)
    ev = _probe("LinearProbeEvaluator", dataset=ds, embedding_cache=str(tmp_path / "emb"))
    with pytest.raises(ValueError, match="before 2026-11-02"):
        ev._load_or_extract(ds)


# ---------------------------------------------------------------------------
# Base-code fixes of 2026-11-02 (approved by Frank Li).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("real", [False, True])
def test_embedding_cache_takes_the_labels_of_the_dataset(tmp_path, real):
    """Q-6 (D-41): a run with the cache uses the labels of the dataset now, not the cached labels.

    Example: a label fix after the cache was made. Here every label is flipped.
    """
    import pickle

    from _toy_eval import ToyClsDS

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 6)).astype(np.float32)
    y = np.stack([(x[:, 0] > 0), (x[:, 1] > 0)], 1).astype(np.float32)
    y[0, 1] = np.nan  # not mentioned (D-05)
    make = (lambda lab: toy_real_cls_ds(x, lab, ["a", "b"])) if real else (lambda lab: ToyClsDS(x, lab, ["a", "b"]))
    cache = str(tmp_path / "emb")
    _probe("LinearProbeEvaluator", dataset=make(y), embedding_cache=cache)._load_or_extract(make(y))
    flipped = make(1.0 - y)
    feats, labels = _probe("LinearProbeEvaluator", dataset=flipped, embedding_cache=cache)._load_or_extract(flipped)
    fresh_feats, fresh = _probe("LinearProbeEvaluator", dataset=flipped)._load_or_extract(flipped)
    np.testing.assert_array_equal(labels, 1.0 - y)
    np.testing.assert_array_equal(labels, fresh)
    np.testing.assert_allclose(feats, fresh_feats, rtol=1e-6)
    with open(f"{cache}.pkl", "rb") as f:
        # Features only, with the inputs that set them (D-41, V-E-34). No labels.
        assert set(pickle.load(f)) == {"feats", "image_paths", "encoder", "base_image_dir", "autocast_dtype"}


def test_embedding_cache_sees_every_change_of_the_features(tmp_path):
    """V-E-25 (D-41): another pool, transform, function encoder or later weight raises.

    The same encoder reads the cache. The cases (a), (b), (d) and (e) are from the proof script
    ``repro_2.py`` of the 2026-10-31 review. Its case (c) is Q-6.
    """
    from radharmony.evaluator.wrappers import ImageEncoderWrapper

    rng = np.random.default_rng(0)
    n, t, d = 20, 5, 8
    tokens = rng.normal(size=(n, t * d)).astype(np.float32)
    labels = (rng.random((n, 1)) > 0.5).astype(np.float32)

    class Tok(torch.nn.Module):
        """[B, T*D] -> [B, T, D] with 2 layers, so the first weight is not all of the model."""

        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.first, self.last = torch.nn.Linear(d, d), torch.nn.Linear(d, d)

        def forward(self, x):
            return self.last(self.first(x.view(x.shape[0], t, d)))

    def run(encoder, ds, name):
        return LinearProbeEvaluator(encoder, dataset=ds, embedding_cache=str(tmp_path / name),
                                    **CPU_KW)._load_or_extract(ds)

    ds, model = toy_real_cls_ds(tokens, labels, ["a"]), Tok()
    run(ImageEncoderWrapper(model, pool=lambda o: o[:, 0]), ds, "w")
    run(ImageEncoderWrapper(model, pool=lambda o: o[:, 0]), ds, "w")  # the same encoder reads the cache
    other = Tok()
    with torch.no_grad():
        other.last.weight.add_(1.0)  # (e) the same first weight
    changes = [
        (ImageEncoderWrapper(model, pool=lambda o: o.mean(dim=1)), ds),  # (a) another pool
        # (b) another transform: the toy transform holds the pixels
        (ImageEncoderWrapper(model, pool=lambda o: o[:, 0]), toy_real_cls_ds(tokens * 2.0, labels, ["a"])),
        (ImageEncoderWrapper(other, pool=lambda o: o[:, 0]), ds),  # (e) another later weight
    ]
    for encoder, data in changes:
        with pytest.raises(ValueError, match="comes from"):
            run(encoder, data, "w")
    run(lambda x: x.view(x.shape[0], -1)[:, :d], ds, "f")
    with pytest.raises(ValueError, match="comes from"):  # (d) 2 function encoders
        run(lambda x: x.view(x.shape[0], -1)[:, d:2 * d], ds, "f")


def test_an_embedding_cache_from_before_the_fingerprint_raises(tmp_path):
    """V-E-25 (D-41): a file with the old encoder name says that it is old, not that the encoder changed."""
    import pickle

    ds = _by_feature(10, ["a"], np.random.default_rng(0))
    rows = [f"img_{i}.png" for i in range(10)]
    with open(tmp_path / "emb.pkl", "wb") as f:
        pickle.dump({"feats": ds.feats, "image_paths": rows, "encoder": "Linear:42:0.0134987"}, f)
    with pytest.raises(ValueError, match="before 2026-11-02"):
        _probe("LinearProbeEvaluator", dataset=ds, embedding_cache=str(tmp_path / "emb"))._load_or_extract(ds)


def test_an_embedding_cache_with_no_image_folder_or_autocast_dtype_raises(tmp_path):
    """V-E-34 (D-41): a file from before 2026-11-03 has no image folder and no autocast_dtype. It says so."""
    import pickle

    ds = _by_feature(10, ["a"], np.random.default_rng(0))
    cache = str(tmp_path / "emb")
    _probe("LinearProbeEvaluator", dataset=ds, embedding_cache=cache)._load_or_extract(ds)
    with open(f"{cache}.pkl", "rb") as f:
        payload = pickle.load(f)
    with open(f"{cache}.pkl", "wb") as f:  # the file as the code of 2026-11-02 wrote it
        pickle.dump({k: payload[k] for k in ("feats", "image_paths", "encoder")}, f)
    with pytest.raises(ValueError, match="before 2026-11-03"):
        _probe("LinearProbeEvaluator", dataset=ds, embedding_cache=cache)._load_or_extract(ds)


_HEAD = torch.nn.Linear(4, 4)  # a global model that a function encoder uses


def test_the_encoder_fingerprint_sees_every_model_and_survives_a_run(tmp_path):
    """V-E-25 (D-41): models in a closure, a partial, a method or a global count. A run keeps the fingerprint."""
    import functools

    from radharmony.dataset.transforms import RadiologyTransform2D
    from radharmony.evaluator.classification.base import _encoder_name
    from radharmony.evaluator.wrappers import ImageEncoderWrapper

    torch.manual_seed(0)
    a, b, body = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)

    def run(model, x):
        return model(x)

    pairs = [
        (ImageEncoderWrapper(body, pool=lambda o: a(o)), ImageEncoderWrapper(body, pool=lambda o: b(o))),
        (functools.partial(run, a), functools.partial(run, b)),
        (a.forward, b.forward),
    ]
    for one, other in pairs:
        assert _encoder_name(one) != _encoder_name(other)
    before = _encoder_name(lambda x: _HEAD(x))
    with torch.no_grad():
        _HEAD.weight.add_(1.0)
    assert _encoder_name(lambda x: _HEAD(x)) != before  # a global model counts

    # The random steps keep their last draw, so only the steps before them count (D-15).
    from PIL import Image

    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(tmp_path / "a.png")
    t = RadiologyTransform2D(img_size=8, output_keys={"img"}, dtype=torch.float32).with_flip(prob=1.0)
    t = t.with_intensity_jitter(shift=0.2, prob=1.0).get_transform()
    first = _encoder_name(body, t)
    t({"img": str(tmp_path / "a.png")})  # a run draws new random values
    assert _encoder_name(body, t) == first


@pytest.mark.parametrize("kind", ["quantized", "extra_state", "meta", "sparse"])
def test_the_encoder_fingerprint_takes_any_state_dict(kind):
    """V-E-25 (D-41): a quantized, meta or sparse tensor, or extra state, gives a fingerprint and no crash."""
    from radharmony.evaluator.classification.base import _encoder_name

    class Extra(torch.nn.Linear):
        def get_extra_state(self):
            return {"note": "x", "scale": torch.tensor([1.0, 2.0])}

        def set_extra_state(self, state):
            pass

    if kind == "quantized":
        model = torch.ao.quantization.quantize_dynamic(torch.nn.Sequential(torch.nn.Linear(4, 4)), dtype=torch.qint8)
    elif kind == "extra_state":
        model = Extra(4, 4)
    elif kind == "meta":
        model = torch.nn.Linear(4, 4, device="meta")
    else:
        model = torch.nn.Linear(4, 4)
        model.register_buffer("mask", torch.eye(4).to_sparse())
    assert _encoder_name(model).startswith(type(model).__name__ + ":")


def test_the_encoder_fingerprint_is_the_same_in_every_process():
    """V-E-25 (D-41): no memory address and no hash-seed order in the fingerprint."""
    import os
    import subprocess
    import sys

    code = ("import torch; from radharmony.evaluator.classification.base import _encoder_name; "
            "from radharmony.evaluator.wrappers import ImageEncoderWrapper; torch.manual_seed(0); "
            "e = ImageEncoderWrapper(torch.nn.Linear(4, 4), pool=lambda o: o if o.dtype in {'a', 'b', 'c'} else o); "
            "print(_encoder_name(e))")
    names = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONHASHSEED": seed}).stdout.split()[-1] for seed in ("1", "2", "3")}
    assert len(names) == 1


def test_summary_std_and_ci_need_2_values(tmp_path):
    """V-E-31 (D-19): with fewer than 2 values, the std and the CI are NaN, not 0.0. 3 values are as before."""
    def row(label, seed, value):
        return {"label": label, "n_train": -1, "fold": -1, "seed": seed, "bootstrap": -1, "auroc": value}

    vals = [0.8, 0.9, 0.7]
    df = pd.DataFrame([row("a", 0, 0.884), row("b", 0, 0.7), row("b", 1, np.nan)]
                      + [row("c", s, v) for s, v in enumerate(vals)])
    ev = _probe("LinearProbeEvaluator", dataset=_by_feature(10, ["a"], np.random.default_rng(0)))
    out = pd.read_csv(f"{ev.save_results(df, str(tmp_path))}/results_summary.csv").set_index("label")
    for label in ("a", "b"):
        assert out.loc[label, ["auroc_std", "auroc_ci_lo", "auroc_ci_hi"]].isna().all(), label
    assert out.loc["c", "auroc_std"] == pytest.approx(np.std(vals))
    assert out.loc["c", "auroc_ci_lo"] == pytest.approx(np.percentile(vals, 2.5))
    assert out.loc["a", "auroc_mean"] == pytest.approx(0.884)


def test_summary_counts_the_runs_with_a_value_and_warns_once_per_label(tmp_path):
    """Q-8 (D-19): ``<metric>_n`` counts the runs with a value. A label with NaN runs warns one time."""
    import warnings

    def row(label, fold, f1, mcc):
        return {"label": label, "n_train": -1, "fold": fold, "seed": -1, "bootstrap": -1,
                "auroc": 0.8, "f1": f1, "mcc": mcc}

    df = pd.DataFrame([row("nodule", f, np.nan if f == 0 else 0.5, np.nan if f < 2 else 0.1) for f in range(5)]
                      + [row("edema", f, 0.6, 0.2) for f in range(5)])
    ev = _probe("LinearProbeEvaluator", dataset=_by_feature(10, ["a"], np.random.default_rng(0)))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        path = ev.save_results(df, str(tmp_path))
    msgs = [str(w.message) for w in rec if "NaN in" in str(w.message)]
    assert msgs == ["Label 'nodule': f1 is NaN in 1 of 5 runs, mcc is NaN in 2 of 5 runs. "
                    "Their means use the other runs (D-19)."]
    out = pd.read_csv(f"{path}/results_summary.csv").set_index("label")
    assert out.loc["nodule", ["auroc_n", "f1_n", "mcc_n"]].tolist() == [5, 4, 3]
    assert out.loc["edema", "f1_n"] == 5
    assert out.loc["nodule", "f1_mean"] == pytest.approx(0.5)


def test_summary_mean_uses_point_rows_and_ci_uses_bootstrap_rows(tmp_path):
    """V-E-15 (D-19): mean = mean of the point estimates, CI = percentiles of the bootstrap rows."""
    point = [{"label": "a", "n_train": -1, "fold": -1, "seed": s, "bootstrap": -1, "auroc": v}
             for s, v in ((0, 0.90), (1, 0.94))]
    boot = [{"label": "a", "n_train": -1, "fold": -1, "seed": s, "bootstrap": b, "auroc": 0.80 + 0.01 * b}
            for s in (0, 1) for b in range(10)]
    df = pd.DataFrame(point + boot)
    ev = _probe("LinearProbeEvaluator", dataset=_by_feature(10, ["a"], np.random.default_rng(0)))
    out = pd.read_csv(f"{ev.save_results(df, str(tmp_path))}/results_summary.csv").iloc[0]
    boot_vals = [r["auroc"] for r in boot]
    assert out["auroc_mean"] == pytest.approx(0.92)
    assert out["auroc_ci_lo"] == pytest.approx(np.percentile(boot_vals, 2.5))
    assert out["auroc_ci_hi"] == pytest.approx(np.percentile(boot_vals, 97.5))


def test_base_has_no_unused_bootstrap_helper():
    """R-code-11 (D-32): the shared bootstrap helper is sampling.bootstrap_indices.

    BaseEvaluator._bootstrap_metrics has no caller, and its two branches are the same.
    """
    from radharmony.evaluator.base import BaseEvaluator

    assert not hasattr(BaseEvaluator, "_bootstrap_metrics")


class _FakeAny:
    """Placeholder dataset for tests that call one evaluator method."""


def test_two_stage_report_scoring_draws_patients(tmp_path):
    """D-32: Stage A writes patient_id, and the Stage B script draws as sampling.bootstrap_indices does."""
    import importlib.util
    import pathlib

    from radharmony.evaluator.sampling import bootstrap_indices

    pids = [f"t{i // 2}" for i in range(10)]
    ev = rev.ReportGenerationEvaluator(lambda imgs: ["text"] * len(imgs), dataset=_ToyReportDS(10, pids),
                                       device="cpu", num_workers=0)
    out = pd.read_parquet(ev.generate_only(str(tmp_path / "pairs.parquet")))
    assert out["patient_id"].tolist() == pids
    script = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "evaluator" / "score_reports_radeval.py"
    spec = importlib.util.spec_from_file_location("score_reports_radeval", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for groups in (pids, None):
        a = list(mod.bootstrap_indices(10, 4, 3, groups))
        b = list(bootstrap_indices(10, 4, 3, groups))
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


_GROUNDED = [
    "[Lungs: Pleura] **small left effusion**. No pneumothorax.",
    "<|ref|> effusion <|/ref|> <|box|> (10,20),(30,40) <|/box|> in the left base.",
]
_CLEANED = ["small left effusion. No pneumothorax.", "effusion in the left base."]


def test_both_report_scoring_paths_clean_the_hypothesis(tmp_path, monkeypatch):
    """Q-6 (D-50): evaluate() and the Stage B script score the same cleaned text. An argument turns it off."""
    import importlib
    import importlib.util
    import pathlib
    import sys
    import types

    from radharmony.evaluator.language import normalize_hypothesis

    assert [normalize_hypothesis(h) for h in _GROUNDED] == _CLEANED
    rg = importlib.import_module("radharmony.evaluator.language.report_generation")
    seen = []
    monkeypatch.setattr(rg, "compute_report_metrics", lambda refs, hyps, metrics: seen.append(hyps) or {"n": 1})

    def gen(imgs):
        return [_GROUNDED[i % 2] for i in range(len(imgs))]

    for kw, expected in (({}, _CLEANED), ({"normalize_hyp": False}, _GROUNDED)):
        seen.clear()
        ev = rev.ReportGenerationEvaluator(gen, dataset=_ToyReportDS(2, ["t0", "t1"]),
                                           num_workers=0, metrics=("n",), **kw)
        ev.evaluate()
        assert seen == [expected]

    # Stage B: a fake RadEval records the hypotheses that it scores.
    pairs = tmp_path / "pairs.parquet"
    pd.DataFrame({"sample_id": ["a", "b"], "reference": ["x", "y"], "hypothesis": _GROUNDED}).to_parquet(pairs)
    stage_b = []

    class _RadEval:
        def __init__(self, metrics):
            pass

        def __call__(self, refs, hyps):
            stage_b.append(list(hyps))
            return {"n": 1.0}

    monkeypatch.setitem(sys.modules, "radeval", types.SimpleNamespace(RadEval=_RadEval))
    script = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "evaluator" / "score_reports_radeval.py"
    spec = importlib.util.spec_from_file_location("score_reports_radeval", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert pathlib.Path(mod.normalize_hypothesis.__code__.co_filename).resolve() == \
        pathlib.Path(rg.__file__).resolve().parent / "hypothesis.py"  # one function, not a copy
    for flag, expected in (([], _CLEANED), (["--no-normalize-hyp"], _GROUNDED)):
        stage_b.clear()
        argv = ["x", "--pairs-parquet", str(pairs), "--output-dir", str(tmp_path / "out"), "--light", *flag]
        monkeypatch.setattr(sys, "argv", argv)
        mod.main()
        assert stage_b == [expected]


def test_report_sample_id_is_the_image_id(tmp_path):
    """V-E-26 (D-46): sample['report'] is the report text, so sample_id comes from image_path, not the report."""
    ev = rev.ReportGenerationEvaluator(lambda imgs: ["text"] * len(imgs),
                                       dataset=_ToyReportDS(4, ["t0", "t0", "t1", "t1"]),
                                       device="cpu", num_workers=0)
    out = pd.read_parquet(ev.generate_only(str(tmp_path / "pairs.parquet")))
    assert out["sample_id"].tolist() == ["img_0", "img_1", "img_2", "img_3"]


def test_seg_val_part_with_no_foreground_has_no_signal():
    """V-E-16: a seg val part with no foreground pixel gives no signal (-inf), also with false positives."""
    from _toy_eval import ToySegDS, toy_seg_data

    imgs, masks = toy_seg_data(n=8)
    val = ToySegDS(imgs, [m * 0 for m in masks])  # no foreground in the ground truth
    ev = rev.LinearProbeSegEvaluator(torch.nn.Identity(), dataset=val, batch_size=4, **CPU_KW)
    head = ev._make_head(4, 8)
    torch.nn.init.constant_(head.conv.bias, 0.0)
    head.conv.bias.data[1] = 5.0  # predicts foreground everywhere
    loader = ev._make_loader(val.get_datasets(), shuffle=False)
    assert ev._val_metric(head, loader) == -float("inf")


def test_frozen_probe_patient_ids_follow_the_dataset_rows():
    """V-E-9, D-32: get_data_dict can drop rows (empty mask_path). The patient ids follow the feature rows."""
    from _toy_eval import _FeatTransform

    from radharmony.dataset.base import BaseRadiologicalDataset

    class _Real(BaseRadiologicalDataset):
        LABEL_COLS: list = []

        def _get_harmonized_df(self):
            return self._try_resolve_preset_harmonized()

    def make(n, drop, rng):
        x = rng.normal(size=(n, 6)).astype(np.float32)
        names = [f"img_{i}.png" for i in range(n)]
        df = pd.DataFrame({"image_path": names, "patient_id": [f"p{i}" for i in range(n)],
                           "mask_path": [None if i in drop else f"m{i}" for i in range(n)],
                           "a": (x[:, 0] > 0).astype(float)})
        t = _FeatTransform({nm: x[i] for i, nm in enumerate(names)}, {f"m{i}": np.zeros((1, 2, 2)) for i in range(n)})
        return _Real(base_image_dir="", transform=t, cache_dir=None, output_cls=True, output_mask=True,
                     harmonized_df=df)

    rng = np.random.default_rng(0)
    ev = _probe("LinearProbeEvaluator", train_dataset=make(60, {1, 2, 3}, rng), test_dataset=make(40, {5}, rng),
                n_bootstrap=2)
    df = ev.evaluate()
    assert sorted(set(df["bootstrap"])) == [-1, 0, 1]


def test_finetune_load_model_uses_the_checkpoint_settings(tmp_path):
    """V-E-19, V-E-20: a loaded head gets the features and score function that it was trained with."""
    import warnings

    from _toy_eval import ToyClsDS

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6)).astype(np.float32) * 5
    ds = ToyClsDS(x, np.eye(2, dtype=np.float32)[rng.integers(0, 2, 40)], ["c0", "c1"])
    ev = rev.FinetuneEvaluator(torch.nn.Identity(), dataset=ds, loss="ce", freeze_backbone=True,
                               l2_normalize=False, epochs=1, val_fraction=0.0, **CPU_KW)
    ev.fit()
    path = ev.save_model(str(tmp_path / "ft.pt"))
    ev2 = rev.FinetuneEvaluator(torch.nn.Identity(), dataset=ds, freeze_backbone=True, **CPU_KW)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ev2.load_model(path)
    np.testing.assert_allclose(np.stack([p["prob"] for p in ev2.predict(ds)]),
                               np.stack([p["prob"] for p in ev.predict(ds)]), rtol=1e-5)


def test_masked_bce_is_the_mean_over_usable_labels():
    """D-18: a label with no 0/1 row in the batch does not make the loss smaller."""
    import torch.nn.functional as F

    ev = rev.FinetuneEvaluator(torch.nn.Identity(), dataset=_FakeAny(), **CPU_KW)
    logits = torch.tensor([[0.3, 1.0], [-0.5, 2.0]])
    cls = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])  # label b has no usable row
    expected = F.binary_cross_entropy_with_logits(logits[:, 0], cls[:, 0])
    assert torch.allclose(ev._masked_loss(logits, cls), expected)


def test_no_validation_part_gives_nan_threshold_metrics():
    """D-29: with no validation part, only a fixed threshold is known."""
    rng = np.random.default_rng(0)
    train, test = _by_feature(100, ["a"], rng), _by_feature(50, ["a"], rng)
    for strategy, expected in (("youden", np.nan), ("fixed:0.5", 0.5)):
        df = _probe("LinearProbeEvaluator", train_dataset=train, test_dataset=test,
                    val_fraction=0.0, threshold_strategy=strategy).evaluate()
        row = df[df["label"] == "a"].iloc[0]
        np.testing.assert_equal(row["threshold"], expected)
        assert np.isfinite(row["auroc"]) and np.isnan(row["f1"]) == np.isnan(expected)


@pytest.mark.parametrize("name", _FIXED_SPLIT_PROBES)
def test_fixed_split_labels_pair_by_name(name):
    """V-E-18: train and test label columns pair by name, not by position."""
    rng = np.random.default_rng(0)
    train, test = _by_feature(300, ["a", "b"], rng), _by_feature(200, ["b", "a"], rng)
    df = _probe(name, train_dataset=train, test_dataset=test).evaluate()
    point = df[df["bootstrap"] == -1].set_index("label")["auroc"]
    assert point["a"] > 0.8 and point["b"] > 0.8, point.to_dict()


@pytest.mark.parametrize("name", _FIXED_SPLIT_PROBES + ["LinearProbeSegEvaluator"])
def test_kfold_n_train_keeps_the_requested_value(name):
    """V-E-21: n_train larger than a fold's train pool keeps the requested value in every fold."""
    from _toy_eval import ToySegDS, toy_seg_data

    rng = np.random.default_rng(0)
    if name == "LinearProbeSegEvaluator":
        ds = ToySegDS(*toy_seg_data(n=21))  # fold pools of 14 and 14 or 13 rows
        ev = rev.LinearProbeSegEvaluator(torch.nn.Identity(), dataset=ds, n_folds=3, epochs=1,
                                         batch_size=4, n_train_samples=[1000], **CPU_KW)
    else:
        ds = _by_feature(103, ["a"], rng)  # fold pools of 82 and 83 rows
        ev = _probe(name, dataset=ds, n_folds=5, n_train_samples=[1000])
    df = ev.evaluate()
    assert set(df["n_train"]) == {1000}


@pytest.mark.parametrize("mode", ["kfold", "fixed"])
@pytest.mark.parametrize("name", ["FinetuneEvaluator", "LinearProbeSegEvaluator", "ConvProbeSegEvaluator"])
def test_same_base_seed_gives_same_result(name, mode):
    """V-E-4, R-code-3: the same base_seed gives the same numbers (torch RNG is seeded)."""
    import pandas as pd

    import copy

    from _toy_eval import ToySegDS, toy_seg_data

    encoder = torch.nn.Linear(6, 6)

    def run():
        torch.rand(100)  # moves the global torch RNG, as other code would
        if name == "FinetuneEvaluator":
            rng = np.random.default_rng(0)
            data = {"dataset": _by_feature(60, ["a"], rng)} if mode == "kfold" else \
                {"train_dataset": _by_feature(60, ["a"], rng), "test_dataset": _by_feature(30, ["a"], rng)}
            ev = rev.FinetuneEvaluator(copy.deepcopy(encoder), n_folds=2, epochs=2, batch_size=8,
                                       store_final_model=True, **data, **CPU_KW)
            df = ev.evaluate()
            return df, [p.detach().clone() for p in ev.final_head_.parameters()]
        data = {"dataset": ToySegDS(*toy_seg_data(n=16))} if mode == "kfold" else \
            {"train_dataset": ToySegDS(*toy_seg_data(n=16)), "test_dataset": ToySegDS(*toy_seg_data(n=8, seed=1))}
        ev = getattr(rev, name)(torch.nn.Identity(), n_folds=2, epochs=2, batch_size=4,
                                store_final_model=True, **data, **CPU_KW)
        df = ev.evaluate()
        return df, [p.detach().clone() for p in ev.final_head_.parameters()]

    (df1, w1), (df2, w2) = run(), run()
    pd.testing.assert_frame_equal(df1, df2)
    assert all(torch.equal(a, b) for a, b in zip(w1, w2, strict=True))


@pytest.mark.parametrize("name", _FIXED_SPLIT_PROBES)
def test_fixed_split_different_label_sets_raise(name):
    """V-E-18: with labels=None and different label sets, the probe raises (no silent pairing)."""
    rng = np.random.default_rng(0)
    train, test = _by_feature(300, ["a", "b", "c"], rng), _by_feature(200, ["a", "c"], rng)
    with pytest.raises(ValueError, match="different labels"):
        _probe(name, train_dataset=train, test_dataset=test).evaluate()
    df = _probe(name, train_dataset=train, test_dataset=test, labels=["a", "c"]).evaluate()
    assert set(df["label"]) == {"a", "c", "macro_average"}


# ---------------------------------------------------------------------------
# Fixes of 2026-11-03 (D-32, D-41). V-E-33 and Q-19 are base code, approved by Frank Li.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", _FIXED_SPLIT_PROBES + ["ZeroShotEvaluator"])
def test_a_table_with_no_patient_id_column_draws_images(name):
    """V-E-33 (D-32): with no ``patient_id`` column, every classification evaluator runs.

    D-32 says that the bootstrap then draws images. So the expected result is the result of the
    same table where each image is its own patient. The k-fold split and the validation part
    also treat each image as a patient.
    """
    from _toy_eval import ToyClsDS

    rng = np.random.default_rng(0)
    train, test = _by_feature(60, ["a"], rng), _by_feature(40, ["a"], rng)

    def ds(t, column):
        d = ToyClsDS(t.feats, t.labels, ["a"], [f"q{i}" for i in range(len(t.feats))])  # one patient per image
        if not column:
            d.df = d.df.drop(columns="patient_id")
        return d

    def run(column):
        if name == "ZeroShotEvaluator":
            return [rev.ZeroShotEvaluator(torch.nn.Identity(), lambda texts: torch.eye(6)[: len(texts)], {"a": ["a"]},
                                          test_dataset=ds(test, column), n_bootstrap=3, **CPU_KW).evaluate()]
        fixed = _probe(name, train_dataset=ds(train, column), test_dataset=ds(test, column), n_bootstrap=3,
                       val_fraction=0.2).evaluate()
        return [fixed, _probe(name, dataset=ds(train, column), n_folds=2, val_fraction=0.2).evaluate()]

    got = run(False)
    for one, want in zip(got, run(True), strict=True):
        pd.testing.assert_frame_equal(one, want)
    assert sorted(set(got[0]["bootstrap"])) == [-1, 0, 1, 2]


@pytest.mark.parametrize("name", _FIXED_SPLIT_PROBES + ["ZeroShotEvaluator"])
def test_a_nan_patient_id_raises_with_the_count_and_the_image_paths(name):
    """Q-5 (D-32): a NaN ``patient_id`` raises ValueError with the number of such rows and their first images.

    Before, the bootstrap raised TypeError and ``GroupKFold`` raised "Input contains NaN".
    """
    from _toy_eval import ToyClsDS

    from radharmony.evaluator.sampling import patient_ids_of

    rng = np.random.default_rng(0)
    t = _by_feature(40, ["a"], rng)
    bad = ToyClsDS(t.feats, t.labels, ["a"], [f"p{i}" for i in range(36)] + [np.nan] * 4)
    message = r"4 rows have no patient_id.*img_36\.png.*img_37\.png"
    with pytest.raises(ValueError, match=message):
        patient_ids_of(bad)
    with pytest.raises(ValueError, match=message):
        if name == "ZeroShotEvaluator":
            rev.ZeroShotEvaluator(torch.nn.Identity(), lambda texts: torch.eye(6)[: len(texts)], {"a": ["a"]},
                                  test_dataset=bad, n_bootstrap=2, **CPU_KW).evaluate()
        else:
            _probe(name, train_dataset=_by_feature(60, ["a"], rng), test_dataset=bad, n_bootstrap=2).evaluate()
    if name != "ZeroShotEvaluator":
        with pytest.raises(ValueError, match=message):
            _probe(name, dataset=bad, n_folds=2).evaluate()


@pytest.mark.parametrize("name", _FIXED_SPLIT_PROBES + ["ZeroShotEvaluator"])
def test_a_frozen_feature_evaluator_warns_for_a_random_step(name):
    """Q-19 (D-41): a frozen-feature evaluator warns when the transform has a random step.

    It extracts the features once, so each image keeps one random draw. Fine-tune draws again in
    each epoch, so it does not warn. A transform with no random step gives no warning.
    """
    import warnings

    import monai.transforms as mt

    class _Rand(mt.RandomizableTransform):
        """A random step that keeps the sample (MONAI ``RandomizableTrait``)."""

        def __call__(self, d):
            self.randomize(None)
            return d

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6)).astype(np.float32)
    x[:, 3] += 3.0
    y = (x[:, :1] > 0).astype(np.float32)

    def warned(random_step):
        ds = toy_real_cls_ds(x, y, ["a"])
        if random_step:
            ds.transform = mt.Compose([ds.transform, _Rand()])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if name == "ZeroShotEvaluator":
                rev.ZeroShotEvaluator(torch.nn.Identity(), lambda texts: torch.eye(6)[: len(texts)], {"a": ["a"]},
                                      test_dataset=ds, **CPU_KW).evaluate()
            else:
                _probe(name, dataset=ds, n_folds=2).evaluate()
        return [str(w.message) for w in caught if "random step" in str(w.message)]

    assert warned(False) == []
    if name == "FinetuneEvaluator":
        assert warned(True) == []
    else:
        assert warned(True) and "_Rand" in warned(True)[0]


def test_the_random_step_check_keeps_the_random_state():
    """Q-19 (D-41): the check builds no new Compose, so the seed of a random step stays.

    A new MONAI Compose gives each random step a new seed. Then a seeded run is not repeatable.
    """
    import monai.transforms as mt

    from radharmony.evaluator.classification.base import _random_steps

    def draws(check):
        flip = mt.RandFlipd(keys="img", prob=0.5)
        t = mt.Compose([mt.Identityd(keys="img"), mt.Compose([flip])])
        t.set_random_state(seed=0)
        if check:
            assert _random_steps(t) == _random_steps([t]) == ["RandFlipd"]
        return flip.R.rand(3)

    np.testing.assert_array_equal(draws(True), draws(False))


def test_seg_probe_takes_the_patient_ids_of_the_classification_probes():
    """V-E-33, Q-5 (D-32): with no patient_id column, each image is its own patient. A NaN id raises."""
    from _toy_eval import ToySegDS, toy_seg_data

    from radharmony.evaluator.segmentation.linear_probe import LinearProbeSegEvaluator

    imgs, masks = toy_seg_data(n=4)
    ds = ToySegDS(imgs, masks)
    ds.df = ds.df.drop(columns="patient_id")
    assert list(LinearProbeSegEvaluator._patient_ids(ds)) == ["img_0.png", "img_1.png", "img_2.png", "img_3.png"]
    ds = ToySegDS(imgs, masks, patient_ids=["p0", None, "p2", "p3"])
    with pytest.raises(ValueError, match="1 rows have no patient_id"):
        LinearProbeSegEvaluator._patient_ids(ds)


@pytest.mark.parametrize("cls_path", ["linear_probe.LinearProbeSegEvaluator", "conv_probe.ConvProbeSegEvaluator",
                                      "upernet_probe.UPerNetSegEvaluator"])
def test_seg_threshold_strategy_warns_that_it_is_not_used(cls_path):
    """Decided by Frank Li (2026-11-04): the head gives each pixel its highest class, so a set threshold warns."""
    import importlib
    import warnings

    from _toy_eval import ToySegDS, toy_seg_data

    module, name = cls_path.split(".")
    cls = getattr(importlib.import_module(f"radharmony.evaluator.segmentation.{module}"), name)
    ds = ToySegDS(*toy_seg_data(n=4))
    with pytest.warns(UserWarning, match="threshold_strategy is not used"):
        cls(lambda x: x, dataset=ds, device="cpu", threshold_strategy="f1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cls(lambda x: x, dataset=ds, device="cpu")


def test_report_device_warns_that_it_is_not_used():
    """Decided by Frank Li (2026-11-04): the generator of the recipe sets its own device, so a set device warns."""
    import warnings

    with pytest.warns(UserWarning, match="device is not used"):
        rev.ReportGenerationEvaluator(lambda imgs: ["x"] * len(imgs), dataset=_ToyReportDS(2, ["t0", "t1"]),
                                      device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rev.ReportGenerationEvaluator(lambda imgs: ["x"] * len(imgs), dataset=_ToyReportDS(2, ["t0", "t1"]))


def test_finetune_with_a_plain_function_raises_type_error_at_once():
    """Decided by Frank Li (2026-11-04): fine-tune moves and trains the encoder, so it needs an nn.Module."""
    from _toy_eval import ToyClsDS

    x, y = torch.randn(8, 4), torch.randint(0, 2, (8, 1)).float()
    for frozen in (True, False):
        with pytest.raises(TypeError, match="nn.Module"):
            rev.FinetuneEvaluator(lambda imgs: imgs, dataset=ToyClsDS(x, y, ["a"]), freeze_backbone=frozen)
    rev.FinetuneEvaluator(torch.nn.Identity(), dataset=ToyClsDS(x, y, ["a"]), freeze_backbone=True)


class _FakeSiglipProcessor:
    """A SigLIP-like processor: one token for each word, model length 4."""

    image_processor = object()

    def __init__(self):
        self.tokenizer = type("Tok", (), {"model_max_length": 4})()
        self.calls = []

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        self.calls.append({"padding": padding, "truncation": truncation})
        ids = [[1] * len(t.split()) for t in text]
        if truncation:
            ids = [i[: self.tokenizer.model_max_length] for i in ids]
        if padding == "max_length":
            ids = [i + [0] * (self.tokenizer.model_max_length - len(i)) for i in ids]
        return {"input_ids": ids}


_FakeSiglipProcessor.__name__ = "SiglipProcessorFake"


def test_zero_shot_cuts_a_long_siglip_prompt_and_warns():
    """Decided by Frank Li (2026-11-04): a prompt over the model length is cut, as the HF pipeline does, with a warning."""
    proc = _FakeSiglipProcessor()
    seen = []

    def text_encoder(tokens):
        seen.append([len(i) for i in tokens["input_ids"]])
        return torch.ones(len(tokens["input_ids"]), 3)

    ev = rev.ZeroShotEvaluator(torch.nn.Identity(), text_encoder, {"a": ["a b"]}, tokenizer=proc,
                               dataset=_ToyReportDS(2, ["t0", "t1"]), device="cpu")
    with pytest.warns(UserWarning, match="1 prompt.*longer than 4 tokens"):
        ev._encode_text(["a b", "a b c d e f"])
    assert seen == [[4, 4]]
    assert proc.calls[-1] == {"padding": "max_length", "truncation": True}
