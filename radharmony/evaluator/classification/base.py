"""Base class for vision classification evaluators.

Adds the frozen image encoder, per-split embedding cache, and the shared
``extract_embeddings`` routine on top of :class:`BaseEvaluator`.
"""

from __future__ import annotations

import dis
import functools
import hashlib
import json
import os
import pickle
import re
import types
import warnings
from contextlib import nullcontext

import monai.transforms as mt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ...dataset.base import _code_hash, _describe, _transform_key
from ..base import BaseEvaluator
from ..sampling import dataset_rows, group_ids


def _value_bytes(value) -> bytes:
    """Return the bytes of one ``state_dict`` entry: a tensor of any kind, or extra state (V-E-25)."""
    if isinstance(value, (list, tuple)):
        return b"(" + b",".join(_value_bytes(v) for v in value) + b")"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return b"{" + b",".join(str(k).encode() + b":" + _value_bytes(v) for k, v in items) + b"}"
    if not isinstance(value, torch.Tensor):
        return json.dumps(_describe(value), sort_keys=True, default=str).encode()
    if value.is_meta:  # no data, only the shape
        return f"meta{tuple(value.shape)}{value.dtype}".encode()
    t = value.detach()
    t = t.dequantize() if t.is_quantized else t
    t = t.to_dense() if t.layout != torch.strided else t
    return t.cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()


def _weights(model, seen: dict) -> str:
    """Return a short hash of every weight and buffer of ``model``. ``seen`` hashes each model once."""
    if id(model) not in seen:
        h = hashlib.sha1()
        for name, value in model.state_dict().items():
            h.update(name.encode())
            h.update(_value_bytes(value))
        seen[id(model)] = h.hexdigest()[:16]
    return seen[id(model)]


def _global_loads(code) -> set:
    """Return the global names that the code loads. Attribute names, as ``net.model``, are not in it."""
    names = {ins.argval for ins in dis.get_instructions(code) if ins.opname == "LOAD_GLOBAL"}
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_loads(const)
    return names


def _settings(obj, seen: dict, depth: int = 0):
    """Describe an encoder for its fingerprint: its code, its settings and the weights of its models.

    Each torch model gives the hash of its weights. A model from outside RadHarmony also gives
    its functions, but not its other attributes. They can change when it runs, for example
    ``warnings_issued`` of a Hugging Face model.
    """
    if depth > 6:
        return type(obj).__name__
    if isinstance(obj, torch.nn.Module):
        ours = type(obj).__module__.startswith("radharmony")
        own = {k: v for k, v in vars(obj).items() if not k.startswith("_") and k != "training"
               and (ours or callable(v))}
        return {"type": type(obj).__qualname__, "weights": _weights(obj, seen),
                **{k: _settings(v, seen, depth + 1) for k, v in sorted(own.items())}}
    if isinstance(obj, functools.partial):
        return {"partial": _settings(obj.func, seen, depth + 1), "args": _settings(list(obj.args), seen, depth + 1),
                "keywords": _settings(dict(obj.keywords), seen, depth + 1)}
    if isinstance(obj, (types.FunctionType, types.MethodType)):
        fn = getattr(obj, "__func__", obj)
        models = {n: fn.__globals__[n] for n in sorted(_global_loads(fn.__code__))
                  if isinstance(fn.__globals__.get(n), torch.nn.Module)}  # a lambda can use a global model
        desc = {"fn": f"{fn.__module__}.{fn.__qualname__}", "code": _code_hash(fn.__code__, fn.__doc__),
                "defaults": _settings(list(fn.__defaults__ or ()), seen, depth + 1),
                "closure": [_settings(c.cell_contents, seen, depth + 1) for c in (fn.__closure__ or ())],
                "globals": _settings(models, seen, depth + 1)}
        if isinstance(obj, types.MethodType):
            desc["self"] = _settings(obj.__self__, seen, depth + 1)
        return desc
    if isinstance(obj, (list, tuple)):
        return [_settings(v, seen, depth + 1) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _settings(v, seen, depth + 1) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if callable(obj) and hasattr(obj, "__dict__") and not isinstance(obj, type):  # a callable object
        call = type(obj).__call__
        own = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {"type": type(obj).__qualname__,
                "call": _code_hash(call.__code__, call.__doc__) if isinstance(call, types.FunctionType) else None,
                **{k: _settings(v, seen, depth + 1) for k, v in sorted(own.items())}}
    return _describe(obj, depth + 1)


def _encoder_name(encoder, transform=None) -> str:
    """Return a fingerprint of the encoder and the image transform, for the embedding cache (V-E-2, V-E-25).

    It hashes the weights and buffers of each torch model in the encoder. It also hashes the
    code and settings of the encoder, for example its pool function, and the cache key of the
    transform (D-15): the steps before the first random step. A random step changes its state
    when it runs. Example: ``RadDino:3f9a1c0b7e2d4a11``.
    """
    text = json.dumps(_settings(encoder, {}), sort_keys=True, default=str)
    if transform is not None:
        text += "|" + _transform_key(transform)
    model = getattr(encoder, "model", encoder)
    return f"{type(model).__name__}:{hashlib.sha1(text.encode()).hexdigest()[:16]}"


def _random_steps(transform) -> list[str]:
    """Return the class names of the random steps (MONAI ``RandomizableTrait``) in ``transform`` (Q-19).

    A nested ``Compose`` is opened. Example: ``Compose([LoadImaged, RandFlipd])`` gives ``["RandFlipd"]``.
    It builds no new ``Compose``: a new ``Compose`` gives each random step a new seed.
    """
    if isinstance(transform, (list, tuple)):
        steps = transform
    elif type(transform) is mt.Compose:  # OneOf and SomeOf are random steps themselves
        steps = transform.transforms
    else:
        return [type(transform).__name__] if isinstance(transform, mt.RandomizableTrait) else []
    return [name for s in steps for name in _random_steps(s)]


class BaseClsEvaluator(BaseEvaluator):
    """Base for the four classification evaluators (linear probe, kNN, zero-shot, finetune).

    The encoder contract is intentionally minimal: ``image_encoder`` is any
    ``nn.Module`` / callable with ``forward(imgs) -> Tensor[B, D]``. The
    evaluator does not look at the type or internals of the encoder. The user
    must wrap HuggingFace/timm/CLIP models into this shape.
    """

    def __init__(
        self,
        image_encoder,
        *,
        dataset=None,
        train_dataset=None,
        test_dataset=None,
        labels: list[str] | None = None,
        device: str = "cuda",
        batch_size: int = 64,
        num_workers: int = 4,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        embedding_cache: str | None = None,
        l2_normalize: bool = False,
        output_dir: str | None = None,
        n_seeds: int = 1,
        base_seed: int = 0,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        threshold_strategy: str = "youden",
    ):
        super().__init__(
            dataset=dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_dir=output_dir,
            n_seeds=n_seeds,
            base_seed=base_seed,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            threshold_strategy=threshold_strategy,
        )
        self.image_encoder = image_encoder
        self.labels = labels
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.autocast_dtype = autocast_dtype
        self.embedding_cache = embedding_cache
        self.l2_normalize = l2_normalize

        # Set in the first extract_embeddings call. It keeps the sorted order of
        # LABEL_COLS (see _sync_label_cols_from_harmonizer in radharmony/dataset/base.py).
        self._active_labels: list[str] | None = None
        self._label_idx: np.ndarray | None = None
        # One group id per sample, in DataLoader order: the patient_id, or the image_path
        # with no patient_id column (D-32). Used for GroupKFold and the bootstrap.
        self._patient_ids: np.ndarray | None = None

    def _resolve_label_indices(self, all_labels: list[str]) -> np.ndarray:
        """Return indices into *all_labels* corresponding to self.labels (case-insensitive)."""
        if self.labels is None:
            self._active_labels = list(all_labels)
            return np.arange(len(all_labels))

        lower = [s.lower() for s in all_labels]
        idxs = []
        resolved = []
        for name in self.labels:
            try:
                i = lower.index(name.lower())
            except ValueError:
                raise KeyError(
                    f"Label {name!r} not found in dataset LABEL_COLS {all_labels}"
                ) from None
            idxs.append(i)
            resolved.append(all_labels[i])
        self._active_labels = resolved
        return np.asarray(idxs, dtype=int)

    def _freeze_encoder(self):
        if isinstance(self.image_encoder, torch.nn.Module):
            self.image_encoder.eval()
            self.image_encoder.to(self.device)
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

    def extract_embeddings(self, ds) -> tuple[np.ndarray, np.ndarray]:
        """Extract pooled embeddings + label matrix for a RadHarmony dataset.

        Returns ``(feats [N, D], labels [N, L])`` as numpy arrays, with
        ``L = len(self.labels)`` (or all labels if ``self.labels is None``).
        """
        torch_ds = ds.get_datasets(n_splits=None, num_cores=self.num_workers)
        # LABEL_COLS is sorted in place at radharmony/dataset/base.py:546.
        all_labels = list(ds.LABEL_COLS)

        # One patient id per sample, from the rows that the dataset gives (V-E-9).
        # With no patient_id column, each image is its own patient (D-32, V-E-33).
        self._patient_ids = group_ids(ds)
        idx = self._resolve_label_indices(all_labels)

        loader = DataLoader(
            torch_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self._freeze_encoder()
        cm = (
            torch.autocast(device_type=self._autocast_device, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else nullcontext()
        )

        feats_list, labels_list = [], []
        with torch.no_grad(), cm:
            for batch in tqdm(loader, desc="extracting embeddings", unit="batch"):
                imgs = batch["img"].to(self.device, non_blocking=True)
                feats = self.image_encoder(imgs)
                if isinstance(feats, torch.Tensor):
                    feats = feats.float().detach().cpu().numpy()
                else:
                    raise TypeError(
                        f"image_encoder must return a Tensor [B, D]; got {type(feats).__name__}. "
                        f"Wrap your model in a callable that extracts the pooled feature."
                    )
                cls = batch["cls"]
                if cls.ndim == 1:
                    cls = cls.view(-1, len(all_labels))
                feats_list.append(feats)
                labels_list.append(cls.float().numpy())

        feats_arr = np.concatenate(feats_list, axis=0)
        all_labels_arr = np.concatenate(labels_list, axis=0)
        self._label_idx = idx
        return feats_arr, all_labels_arr[:, idx]

    @property
    def _autocast_device(self) -> str:
        if self.device.startswith("cuda"):
            return "cuda"
        return "cpu"

    def _autocast_cm(self):
        """Return an autocast context with ``self.autocast_dtype``.

        The dataset gives bfloat16 tensors (``RadiologyEncoderTransform`` casts
        the image to bf16 at the end), but the encoder weights are fp32.
        Autocast casts each op, so this works. ``BaseSegEvaluator._autocast_cm``
        does the same.
        """
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self._autocast_device, dtype=self.autocast_dtype
        )

    def _dataset_labels(self, ds) -> np.ndarray:
        """Return the label matrix ``[N, len(ds.LABEL_COLS)]`` of the samples (D-41).

        It reads ``cls`` from the samples before the transform (``.data`` of a MONAI
        dataset), so a MONAI dataset loads no image. These are the values that
        ``extract_embeddings`` gets.
        """
        torch_ds = ds.get_datasets(n_splits=None, num_cores=self.num_workers)
        items = getattr(torch_ds, "data", None)
        items = items if items is not None else [torch_ds[i] for i in range(len(torch_ds))]
        n = len(ds.LABEL_COLS)
        rows = [np.asarray(d["cls"], dtype=np.float32).reshape(n) for d in items]
        return np.stack(rows) if rows else np.zeros((0, n), np.float32)

    def _load_or_extract(
        self, ds, cache_suffix: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings, or load from ``embedding_cache`` if present.

        Caches store **raw** features only. The labels always come from the dataset,
        so a label fix is seen (D-41). L2 normalization (when enabled via
        ``l2_normalize=True``) is applied after load so the same cache file
        is reusable across evaluators with different normalize settings.

        Raises:
            ValueError: the cache file was made before 2026-11-03. Or it has other images,
                another image folder, encoder, transform or ``autocast_dtype`` (V-E-2, V-E-25, V-E-34).

        Warns:
            UserWarning: the transform has a random step (D-41, Q-19).
        """
        rand_steps = _random_steps(getattr(ds, "transform", None))
        if rand_steps:  # fine-tune does not call this: it draws again in each epoch
            warnings.warn(
                f"The transform has a random step ({', '.join(rand_steps)}). A frozen-feature evaluator "
                "extracts the features once, so each image keeps one random draw. An embedding "
                "cache keeps it too (D-41). Use a transform with no random step.",
                UserWarning, stacklevel=3,
            )
        cache_path = None
        if self.embedding_cache is not None:
            if cache_suffix:
                cache_path = f"{self.embedding_cache}.{cache_suffix}.pkl"
            else:
                cache_path = self.embedding_cache
                if not cache_path.endswith(".pkl"):
                    cache_path = cache_path + ".pkl"

        rows = dataset_rows(ds)
        image_paths = rows["image_path"].astype(str).tolist()
        # Hash the weights only when a cache is used: it takes about 1 s for 300 million weights.
        encoder = _encoder_name(self.image_encoder, getattr(ds, "transform", None)) if cache_path else None
        # The same relative paths in another folder are other images, and bf16 and fp32 give other features (V-E-34).
        image_dir = getattr(ds, "base_image_dir", None)
        settings = {"base_image_dir": os.path.abspath(image_dir) if image_dir else "",
                    "autocast_dtype": str(self.autocast_dtype)}
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            # Use a cache only for the same images, encoder and transform (V-E-2, V-E-25).
            if not re.fullmatch(r"[^:]+:[0-9a-f]{16}", str(payload.get("encoder"))):
                raise ValueError(
                    f"The embedding cache {cache_path} was made before 2026-11-02, so it cannot be "
                    "checked. Delete the file to build it again."
                )
            if not settings.keys() <= payload.keys():
                raise ValueError(
                    f"The embedding cache {cache_path} was made before 2026-11-03, so its image folder "
                    "and autocast_dtype cannot be checked. Delete the file to build it again."
                )
            for key, value in settings.items():
                if payload[key] != value:
                    raise ValueError(
                        f"The embedding cache {cache_path} has {key}={payload[key]!r}, not {value!r}. "
                        "Delete the file, or use another embedding_cache path."
                    )
            if list(payload["image_paths"]) != image_paths:
                raise ValueError(
                    f"The embedding cache {cache_path} has other images than the dataset. "
                    "Delete the file, or use another embedding_cache path."
                )
            if payload["encoder"] != encoder:
                raise ValueError(
                    f"The embedding cache {cache_path} comes from {payload['encoder']}, "
                    f"not {encoder}. Delete the file, or use another embedding_cache path."
                )
            labels = self._dataset_labels(ds)  # the labels of the dataset now, not of the cache (D-41)
            self._label_idx = self._resolve_label_indices(list(ds.LABEL_COLS))
            self._patient_ids = group_ids(ds)
            return self._maybe_normalize(payload["feats"]), labels[:, self._label_idx]

        feats, labels = self.extract_embeddings(ds)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                # Features only: a run reads the labels from the dataset (D-41).
                pickle.dump({"feats": feats, "image_paths": image_paths, "encoder": encoder, **settings}, f)
        return self._maybe_normalize(feats), labels

    def _maybe_normalize(self, feats: np.ndarray) -> np.ndarray:
        if not self.l2_normalize:
            return feats
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms

    @property
    def active_labels(self) -> list[str]:
        if self._active_labels is None:
            raise RuntimeError(
                "active_labels is only available after extract_embeddings() has run."
            )
        return self._active_labels
