
from typing import  Optional, Sequence
from datetime import datetime

import numpy as np
import torch
import zarr


def get_date_based_window_splits(
    timestamps: np.ndarray,  # sorted array of datetimes for the dataset
    periods: list[dict], 
    window_size: int,
    indices_to_avoid: list[int] | None = None,
    exclude_when_end_indices_divisible_by: int | None = None,
) -> dict[str, dict]:
    """
    periods:
      - train:
          start: "2020-01-01"
          end: "2020-12-31"
        val:
          start: "2021-01-01"
          end: "2021-06-30"
        test:
          start: "2021-07-01"
          end: "2021-12-31"
      - train:
          start: "2021-01-01"
          end: "2021-12-31"
        val:
          start: "2022-01-01"
          end: "2022-06-30"
        test:
          start: "2022-07-01"
          end: "2022-12-31"
    """
    results = {
        "train": {"starts": [], "ends": [], "window_size": window_size},
        "val":   {"starts": [], "ends": [], "window_size": window_size},
        "test":  {"starts": [], "ends": [], "window_size": window_size},
    }


    avoid_mask = np.zeros(len(timestamps), dtype=bool)
    if indices_to_avoid is not None:
        avoid_mask[indices_to_avoid] = True


    for period in periods:
        for split_key in ["train", "val", "test"]:
            if split_key not in period:
                continue

            period_start = _parse_dt(period[split_key]["start"])
            period_end   = _parse_dt(period[split_key]["end"])

            mask = (timestamps >= period_start) & (timestamps <= period_end)
            valid_indices = np.where(mask)[0]

            if len(valid_indices) < window_size:
                continue

            idx_start = valid_indices[0]
            idx_end   = valid_indices[-1] + 1

            for win_start in range(idx_start, idx_end - window_size + 1, window_size):
                win_end = win_start + window_size

                if np.any(avoid_mask[win_start:win_end]):
                    continue

                if (
                    exclude_when_end_indices_divisible_by is not None
                    and (win_end - 1) % exclude_when_end_indices_divisible_by == 0
                ):
                    continue

                results[split_key]["starts"].append(win_start)
                results[split_key]["ends"].append(win_end)

    for split_key in ["train", "val", "test"]:
        results[split_key]["starts"] = np.array(results[split_key]["starts"], dtype=np.int32)
        results[split_key]["ends"]   = np.array(results[split_key]["ends"],   dtype=np.int32)

    return results


def _parse_dt(dt) -> np.datetime64:
    if isinstance(dt, np.datetime64):
        return dt.astype("datetime64[s]")
    if isinstance(dt, datetime):
        return np.datetime64(dt, "s")
    if isinstance(dt, str):
        return np.datetime64(datetime.fromisoformat(dt), "s")
    raise TypeError(f"Unsupported date type: {type(dt)}")


def get_count_based_window_splits(
    time_count: int,
    train: int,
    val: int,
    test: int,
    skip: int,
    window_size: int,
    exclude_when_end_indices_divisible_by: int | None = None,

) -> dict[str, np.ndarray]:
    if time_count < window_size * (train + val + test):
        raise ValueError(
            f"Dataset time_count ({time_count}) is smaller than required "
            f"window size ({window_size * (train + val + test)})."
        )

    indices_end = time_count - window_size + 1
    counter = 0

    results = {
        "train": {"starts": [], "ends": [], "window_size": window_size},
        "val":   {"starts": [], "ends": [], "window_size": window_size},
        "test":  {"starts": [], "ends": [], "window_size": window_size},
    }

    def _append(split_key: str, starts: list[int]) -> None:
        for s in starts:
            e = s + window_size
            if (
                exclude_when_end_indices_divisible_by is not None
                and (e - 1) % exclude_when_end_indices_divisible_by == 0
            ):
                continue
            results[split_key]["starts"].append(s)
            results[split_key]["ends"].append(e)

    while counter < indices_end:
        t_limit  = min(counter + train * window_size, indices_end)
        _append("train", list(range(counter, t_limit, window_size)))
        counter += (train + skip) * window_size
        if counter >= indices_end:
            break

        v_limit  = min(counter + val * window_size, indices_end)
        _append("val", list(range(counter, v_limit, window_size)))
        counter += (val + skip) * window_size
        if counter >= indices_end:
            break

        te_limit = min(counter + test * window_size, indices_end)
        _append("test", list(range(counter, te_limit, window_size)))
        counter += (test + skip) * window_size

    for split_key in ["train", "val", "test"]:
        results[split_key]["starts"] = np.array(results[split_key]["starts"], dtype=np.int32)
        results[split_key]["ends"]   = np.array(results[split_key]["ends"],   dtype=np.int32)

    return results

def resolve_var_indices(
    requested: Sequence[str] | None,
    all_vars: Sequence[str],
) -> np.ndarray:
    """Resolve variable indices and selected variable names in final output order."""
    if requested is None:
        return np.arange(len(all_vars), dtype=np.int32)
    var_map = {v: i for i, v in enumerate(all_vars)}
    missing = [v for v in requested if v not in var_map]
    if missing:
        raise KeyError(f"Requested variables not found: {missing}. Available: {list(all_vars)}")
    idx = np.array([var_map[v] for v in requested], dtype=np.int32)
    return idx


def read_var_stat(root: zarr.Group, key: str, idxs: np.ndarray) -> Optional[torch.Tensor]:
    """Read a per-variable 1D stat array from the Zarr group, subset to idx."""
    if key not in root:
        return None
    arr = root[key]  # expected shape: (variable,)
    return torch.as_tensor(arr[idxs], dtype=torch.float32)


def load_stats_from_zarr(root: zarr.Group, idxs: list[int]) -> dict[str, torch.Tensor | None]:
    """
    Load per-variable stats from a Zarr group and return a NormalizerStats object.
    Requires 'minimum' and 'maximum'. Optionally loads 'mean' and ('stdev' or 'std').
    """
    vmin = read_var_stat(root, "minimum", idxs)
    vmax = read_var_stat(root, "maximum", idxs)
    mean = read_var_stat(root, "mean", idxs)
    std = read_var_stat(root, "stdev", idxs)

    if std is None:
        std = read_var_stat(root, "std", idxs)

    return {"minimum": vmin, "maximum": vmax, "mean": mean, "std": std}
