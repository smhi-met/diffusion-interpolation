from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, ClassVar
import zarr
import torch
import numpy as np

from .helpers import get_date_based_window_splits, resolve_var_indices, load_stats_from_zarr, get_count_based_window_splits

from datetime import datetime


@dataclass
class SplitBase:
    """Base class for defining dataset splits."""
    type: str


@dataclass
class CountBasedWindowSplit(SplitBase):
    type: ClassVar[str] = "count_based_window"
    train: int
    val: int
    test: int
    skip: int
    window_size: int


@dataclass
class TimeSlot:
    start: str
    end: str


@dataclass
class Periods:
    train: TimeSlot
    val: TimeSlot
    test: TimeSlot


@dataclass
class DateBasedSplit(SplitBase):
    type: ClassVar[str] = "date_based"
    periods: list[Periods]


@dataclass
class DateBasedWindowSplit(DateBasedSplit):
    type: ClassVar[str] = "date_based_window"
    window_size: int


SplitsConfig = DateBasedSplit | CountBasedWindowSplit | DateBasedWindowSplit | None


class DatasetManifestBase(ABC):
    """Abstract interface enforcing the contract for all dataset providers."""

    @property
    @abstractmethod
    def dataset_path(self) -> str:
        """Returns the universal path to the dataset (e.g., Zarr store path)."""
        pass

    def find_variables_indices(self, variable_list: list[str]) -> np.ndarray:
        """Return indices of the passed variables in the variable set self.variables."""
        return torch.tensor(
            [self.variables.index(v) for v in variable_list],
            dtype=torch.long)

    @cached_property
    @abstractmethod
    def splits(self) -> dict[str, np.ndarray]:
        """Returns the pre-computed train/val/test splits as a dictionary."""
        pass

    @property
    @abstractmethod
    def stats(self) -> dict[str, torch.Tensor]:
        """Returns the normalization statistics dictionary."""
        pass

    @property
    @abstractmethod
    def var_dim(self) -> int:
        """Returns the index of the variable dimension in the dataset tensors."""
        pass

    @property
    @abstractmethod
    def variables(self) -> list[str]:
        """Returns the ordered list of selected variable names."""
        pass


class MepsZarrManifest(DatasetManifestBase):
    """
    Dataset Manifest for MEPS Zarr datasets.
    Handles metadata loading, variable selection, statistics and split generation.

    The ``variables`` parameter accepts two formats:

    Old format (backward compatible)::

        variables: ["10u", "2t", "sp"]
    exclude_when_end_indices_divisible_by: Optional int parameter to exclude windows where the end index is divisible by this value. This can be used to avoid certain time steps that may correspond to data quality issues or other anomalies.
    New format (exposes per-variable normalization config for MultiNormalizer)::

        variables:
          - name: "10u"
            normalization: "z-score"
            scale: 0.5
          - name: "2t"
            normalization: "z-score"
            scale: 1.0

    Use ``manifest.variables_conf`` to retrieve the full per-variable config
    for passing to MultiNormalizer (e.g. via Hydra interpolation
    ``variables_conf: ${manifest.variables}``).
    """

    def __init__(
        self,
        dataset_path: str,
        indices_to_avoid: list[int] | None = None,
        exclude_when_end_indices_divisible_by: int | None = None,
        variables: Sequence[str | dict] | None = None,
        reshape_to: list[int] | None = None,
        splits: SplitsConfig | None = None,
    ):
        self._dataset_path = dataset_path
        self.indices_to_avoid = indices_to_avoid
        self.exclude_when_end_indices_divisible_by = exclude_when_end_indices_divisible_by
        self.reshape_to = reshape_to
        self._splits = splits

        self._variable_names, self._variables_conf = self._parse_variables(variables)

        self._var_indices = slice(None)
        self._stats = {}
        self._time_count = 0
        self._load_metadata()

    @staticmethod
    def _parse_variables(
        variables: Sequence[str | dict] | None,
    ) -> tuple[list[str] | None, list[dict] | None]:
        """
        Normalise the variables input to (names, confs).

        Returns:
            names: ordered list of variable name strings, or None (select all)
            confs: list of full variable config dicts when new dict format is used,
                   None when old string-only format is used (backward compatible)
        """
        if variables is None:
            return None, None

        names: list[str] = []
        confs: list[dict] = []
        has_dict_entries = False

        for v in variables:
            if isinstance(v, str):
                names.append(v)
                confs.append({"name": v})
            elif hasattr(v, "get"):  # dict or Hydra DictConfig
                name = v.get("name")
                if not name:
                    raise ValueError(
                        f"Variable dict entry is missing required 'name' field: {dict(v)}"
                    )
                names.append(name)
                confs.append({k: v[k] for k in v})
                has_dict_entries = True
            else:
                raise ValueError(
                    f"Each variable entry must be a string or a dict with 'name', "
                    f"got {type(v)!r}"
                )

        return names, (confs if has_dict_entries else None)

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @cached_property
    def splits(self) -> dict[str, np.ndarray]:
        """Returns the pre-computed train/val/test splits as a dictionary."""
        if self._splits is None:
            raise ValueError("Split configuration is required to compute dataset splits.")

        if self._splits.type == "date_based":
            if hasattr(self._splits, "window_size"):
                raise ValueError("window_size should not be defined for date_based split type.")
            return get_date_based_window_splits(
                timestamps=self._dates,
                periods=self._splits.periods,
                window_size=1,
                indices_to_avoid=self.indices_to_avoid,
                exclude_when_end_indices_divisible_by=self.exclude_when_end_indices_divisible_by,
            )

        elif self._splits.type == "date_based_window":
            if not hasattr(self._splits, "window_size"):
                raise ValueError("window_size is required for date_based_window split type.")
            if self._splits.window_size < 2:
                raise ValueError("window_size must be greater than 1.")
            return get_date_based_window_splits(
                timestamps=self._dates,
                periods=self._splits.periods,
                window_size=self._splits.window_size,
                indices_to_avoid=self.indices_to_avoid,
                exclude_when_end_indices_divisible_by=self.exclude_when_end_indices_divisible_by,
            )

        elif self._splits.type == "count_based":
            if hasattr(self._splits, "window_size"):
                raise ValueError("window_size should not be defined for count_based split type.")
            return get_count_based_window_splits(
                time_count=self._time_count,
                train=self._splits.train,
                val=self._splits.val,
                test=self._splits.test,
                skip=self._splits.skip,
                window_size=1,
                exclude_when_end_indices_divisible_by=self.exclude_when_end_indices_divisible_by,
            )

        elif self._splits.type == "count_based_window":
            if not hasattr(self._splits, "window_size"):
                raise ValueError("window_size is required for count_based_window split type.")
            if self._splits.window_size < 2:
                raise ValueError("window_size must be greater than 1.")
            return get_count_based_window_splits(
                time_count=self._time_count,
                train=self._splits.train,
                val=self._splits.val,
                test=self._splits.test,
                skip=self._splits.skip,
                window_size=self._splits.window_size,
                exclude_when_end_indices_divisible_by=self.exclude_when_end_indices_divisible_by,
            )

        else:
            raise ValueError(f"Unsupported split type: {self._splits.type}")

    @property
    def stats(self) -> dict[str, torch.Tensor]:
        return self._stats

    @property
    def var_dim(self) -> int:
        if self.reshape_to is not None:
            return -4  # After reshape: (T, V, E, H, W)
        else:
            return -3  # Original: (T, V, E, C)

    @property
    def variables(self) -> list[str] | None:
        return self._variable_names

    @property
    def variables_conf(self) -> list[dict] | None:
        """
        Full per-variable config list (name, normalization, scale).
        None when the old string-only format was used.
        Used by MultiNormalizer via Hydra interpolation: ``${manifest.variables}``.
        """
        return self._variables_conf

    @property
    def variables_indices(self) -> np.ndarray:
        return self._var_indices

    def _load_metadata(self) -> None:
        """Opens Zarr store exactly once to compute indices, stats, and metadata."""
        root = zarr.open(self.dataset_path, mode="r")
        all_vars = list(root.attrs.get("variables", []))
        self.all_vars = all_vars

        if not all_vars:
            raise ValueError("Missing root.attrs['variables']; check Zarr metadata.")

        self._var_indices = resolve_var_indices(self._variable_names, all_vars)
        self._dates = root["dates"][:].astype("datetime64[s]")
        self._stats = load_stats_from_zarr(root, self._var_indices)
        self._time_count = int(root["data"].shape[0])
