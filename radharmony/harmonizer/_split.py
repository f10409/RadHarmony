"""Mixins for the official split classes (D-09): keep one split, fix the split argument, load old files."""

import pickle

import pandas as pd


class _KeepSplitMixin:
    """Keep only the rows of one official split.

    Put this class before the dataset harmonizer in the class list.
    ``_read_source()`` removes the other splits before the base steps. So
    ``preprocess_output_dir`` and ``sample_n`` see only this split (D-30).
    ``harmonize()`` keeps the split again at the end, for a source table
    with no split column.

    Class attributes:
        SPLIT: the value to keep, as in the source (for example ``"val"``).
        SPLIT_COL: the split column of the harmonized table. Default ``"split"``.
        SPLIT_SOURCE_COLS: the names of the split column in the source table,
            in the order to try. TAIX-Ray ``annotation.csv`` has ``Split``.
    """

    SPLIT: str = None
    SPLIT_COL: str = "split"
    SPLIT_SOURCE_COLS: tuple = ("split", "Split")

    def _read_source(self) -> pd.DataFrame:
        """Read the source table and keep the rows of ``SPLIT``.

        A table with no split column stays as it is. ``harmonize()`` then
        keeps the split at the end.
        """
        df = super()._read_source()
        col = next((c for c in self.SPLIT_SOURCE_COLS if c in df.columns), None)
        if col is None:
            return df
        return df[df[col].astype(str) == self.SPLIT].reset_index(drop=True)

    def harmonize(self, *args, **kwargs):
        super().harmonize(*args, **kwargs)
        keep = self.df[self.SPLIT_COL].astype(str) == self.SPLIT
        self.df = self.df[keep].reset_index(drop=True)
        return self._select_harmonized_columns(self.df)


class _FixedSplitArgMixin:
    """For a class that takes ``split=`` (one name) or ``splits=`` (a list): fix it to ``SPLIT``.

    Put this class first in the class list. It works for harmonizers and
    dataset classes. A call with another split raises ``ValueError``, so a
    class named Train cannot load the test set.

    Class attributes:
        SPLIT: the split name, as in the source (for example ``"validation"``).
        SPLIT_ARG: ``"split"`` or ``"splits"``.
    """

    SPLIT: str = None
    SPLIT_ARG: str = "split"

    def __init__(self, *args, **kwargs):
        own = [self.SPLIT] if self.SPLIT_ARG == "splits" else self.SPLIT
        given = kwargs.pop(self.SPLIT_ARG, None)
        if given is not None and self.SPLIT_ARG == "splits" and not isinstance(given, str):
            given = list(given)
        if given is not None and given != own:
            raise ValueError(f"{type(self).__name__} gives only the {self.SPLIT!r} split, not {given!r}.")
        super().__init__(*args, **{self.SPLIT_ARG: own}, **kwargs)


class _LoadOldFileMixin:
    """Also load a file that ``OLD_FILE_CLASS`` saved, so old files still load (D-09).

    Put this class first in the class list of a split harmonizer. The loaded
    object is of the split class. If the class has ``SPLIT``, each value of the
    ``split`` column must be ``SPLIT``. If the table has no ``split`` column,
    the saved arguments must name the split. This is the argument ``split``, or
    an argument that ``OLD_FILE_CLASS._split_of_saved_init(init)`` reads, for
    example the file name ``valid.csv``. If not, it raises ``ValueError``.

    Class attributes:
        OLD_FILE_CLASS: the class that saved the old files, for example the
            class that mixes the splits.
    """

    OLD_FILE_CLASS: type = None

    @classmethod
    def load_from_saved(cls, path: str, **init_overrides):
        old = cls.OLD_FILE_CLASS
        with open(path, "rb") as f:
            payload = pickle.load(f)
        saved = (payload["harmonizer_module"], payload["harmonizer_class"])
        if old is None or saved != (old.__module__, old.__qualname__):
            return super().load_from_saved(path, **init_overrides)
        h = old.load_from_saved(path, **init_overrides)
        df = h.harmonized_df
        split = getattr(cls, "SPLIT", None)
        # When the saved arguments have ``split``, cls(...) below checks it.
        if split is not None and "split" in df.columns:
            other = sorted(set(df["split"].astype(str)) - {split})
            if other:
                raise ValueError(f"{path} has rows of the split {other}. {cls.__name__} gives only {split!r}.")
        elif split is not None and "split" not in payload["init"]:
            named = getattr(old, "_split_of_saved_init", lambda init: None)(payload["init"])
            if named is None:
                raise ValueError(f"{path} has no split column and no split argument, "
                                 f"so {cls.__name__} cannot check that its rows are {split!r}.")
            if named != split:
                raise ValueError(f"{path} is a file of the split {named!r}. {cls.__name__} gives only {split!r}.")
        inst = cls(**{**payload["init"], **init_overrides})
        inst.LABEL_COLS = h.LABEL_COLS
        inst.set_harmonized_df(df)
        return inst


class _OneSplitDatasetMixin:
    """For a dataset class of one official split: ``get_datasets_predefined()`` raises ``TypeError``.

    The class holds the rows of one split, so it has no other split to give. Put this class
    before the dataset class with all splits in the class list. The error names that class.
    """

    def get_datasets_predefined(self, *args, **kwargs):
        mro = type(self).__mro__
        full = mro[mro.index(_OneSplitDatasetMixin) + 1].__name__
        raise TypeError(
            f"{type(self).__name__} holds one split, so it has no get_datasets_predefined(). "
            f"Use {full}(...).get_datasets_predefined() for all splits, or get_datasets() of each split class."
        )
