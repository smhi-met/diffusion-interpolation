import matplotlib.pyplot as plt
import torch

def plot_vars_two_rows(x: torch.Tensor, t_idx: int = 0, titles=None):
    """
    Plot the 5 variable maps at a given time index in a 2-row grid.

    Expects x shaped (T, V, H, W) where V==5.
    Plots x[t_idx, v] for v=0..4 as 2 rows (3 on top, 2 on bottom).
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape (T, V, H, W), got {tuple(x.shape)}")
    T, V, H, W = x.shape
    if not (0 <= t_idx < T):
        raise ValueError(f"t_idx={t_idx} out of range for T={T}")

    n = V
    ncols = 3
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for v in range(n):
        
        ax = axes[v]
        img = x[t_idx, v].detach().cpu().numpy()
        print(f"min: {img.min()}, max: {img.max()}")
        im = ax.imshow(img, origin="lower")
        ax.set_title(titles[v] if titles is not None and v < len(titles) else f"var {v}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused axes (e.g., last slot in 2x3 grid)
    for k in range(n, nrows * ncols):
        axes[k].axis("off")

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import zarr

def plot_dataset_splits_with_borders(splits, splitter, root_or_path=None, data_key="data"):
    """
    splits: dict returned by splitter.get_splits(...)
    splitter: the splitter instance (for window_size/skip in title)
    root_or_path: optional zarr.Group or path to zarr store (used to set x-limits from actual data length)
    data_key: name of the main array in the store (default: "data")
    """
    plt.figure(figsize=(25, 4))

    plot_configs = [
        ("train", "dodgerblue", 0.03),
        ("val",   "darkorange", 0.00),
        ("test",  "forestgreen", -0.03),
    ]

    for key, color, offset in plot_configs:
        split_info = splits.get(key)
        if not split_info or not split_info.get("starts"):
            continue

        starts = np.asarray(split_info["starts"], dtype=int)
        ends = np.asarray(split_info["ends"], dtype=int)

        # Plot internal window indices (as tick marks)
        all_indices = np.concatenate([np.arange(s, e, dtype=int) for s, e in zip(starts, ends)])
        plt.scatter(
            all_indices,
            np.full_like(all_indices, offset, dtype=float),
            color=color,
            alpha=0.3,
            s=100,
            marker="|",
            linewidth=1,
        )

        # Draw window borders + connector bar
        for s, e in zip(starts, ends):
            plt.vlines(
                x=[s - 0.5, e - 0.5],
                ymin=offset - 0.015,
                ymax=offset + 0.015,
                colors="black",
                alpha=0.8,
                linewidth=1.5,
            )
            plt.hlines(y=offset, xmin=s, xmax=e - 1, colors=color, linewidth=8, alpha=0.5)

    plt.title(
        "Windowed Dataset Splits (Index-Based)\n"
        f"Window Size: {splitter.window_size} | Skip: {splitter.skip}",
        fontsize=14,
    )
    plt.xlabel("Original Dataset Index")
    plt.yticks([])

    # X-axis limit:
    # Prefer n_time from zarr root["data"].shape[0] if provided, otherwise infer from splits.
    if root_or_path is not None:
        root = zarr.open(root_or_path, mode="r") if isinstance(root_or_path, (str, bytes)) else root_or_path
        n_time = int(root[data_key].shape[0])
        x_max = n_time - 1
    else:
        x_max = max(max(v["ends"]) for v in splits.values() if v.get("ends")) - 1

    plt.xlim(-10, x_max + 10)
    plt.ylim(-0.1, 0.1)

    legend_elements = [
        Line2D([0], [0], color="dodgerblue", lw=4, label="Train Window"),
        Line2D([0], [0], color="darkorange", lw=4, label="Val Window"),
        Line2D([0], [0], color="forestgreen", lw=4, label="Test Window"),
        Line2D([0], [0], color="black", lw=1.5, label="Sample Border"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.grid(True, axis="x", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage (if you want x-limits from the actual store):
# root = zarr.open(dataset_path, mode="r")
# splits = splitter.get_splits(dataset_path)
#plot_dataset_splits_with_borders(splits, splitter, root_or_path=root)


import matplotlib.pyplot as plt
import numpy as np
import zarr


def _format_epoch_seconds_to_hour_str(epoch_seconds: int) -> str:
    """Convert Unix epoch seconds (int) -> ISO string at hour resolution."""
    dt64 = np.datetime64(int(epoch_seconds), "s")
    return np.datetime_as_string(dt64, unit="h")


def plot_window_comparison_full(root_or_path, split_dict, sample_idx: int = 0, var_idx: int = 2, data_key: str = "data"):
    """
    Zarr-native window comparison plot.

    Parameters
    ----------
    root_or_path : zarr.Group | str
        Open zarr root group or path to zarr store.
    split_dict : dict
        One split entry from BlockWindowSplitter, e.g. splits["train"] with keys
        "starts", "ends", "window_size".
    sample_idx : int
        Which window to plot within the split.
    var_idx : int
        Variable index in the zarr 'variable' dimension.
    data_key : str
        Key for the main zarr array, default "data".
    """
    root = zarr.open(root_or_path, mode="r") if isinstance(root_or_path, (str, bytes)) else root_or_path
    data = root[data_key]  # (time, variable, ensemble, cell)

    start_idx = int(split_dict["starts"][sample_idx])
    window_size = int(split_dict["window_size"])

    var_names = list(root.attrs.get("variables", ["10u", "10v", "2t", "10si", "tp"]))
    var_name = var_names[var_idx] if 0 <= var_idx < len(var_names) else f"var_{var_idx}"

    dates = root["dates"] if "dates" in root else None

    fig, axes = plt.subplots(window_size, 2, figsize=(12, 4 * window_size))
    if window_size == 1:
        axes = np.expand_dims(axes, axis=0)

    for step in range(window_size):
        global_idx = start_idx + step

        # Read 1D field for this timestep/variable (ensemble assumed 0)
        data_1d = data[global_idx, var_idx, 0, :]  # (cell,)

        date_label = ""
        if dates is not None:
            date_str = _format_epoch_seconds_to_hour_str(dates[global_idx])
            # show hour portion if ISO-like
            date_label = f"{date_str.split('T')[1]}h" if "T" in date_str else date_str

        grid_2d = data_1d.reshape(256, 256)

        for i, label in enumerate(["Zarr View", "Index-Mapped View"]):
            ax = axes[step, i]
            im = ax.imshow(grid_2d, cmap="magma", origin="lower")

            if step == 0:
                ax.set_title(f"{label}\nGlobal Index: {global_idx}\nindex: {sample_idx}", fontweight="bold")
            else:
                ax.set_title(f"Global Index: {global_idx}\nindex: {sample_idx}")

            if i == 0:
                ax.set_ylabel(f"Step {step}\n{date_label}", fontsize=12, fontweight="bold")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Window Comparison: {var_name} | Sample: {sample_idx}\nWindow Start: {start_idx}",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
