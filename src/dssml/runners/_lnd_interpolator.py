"""
Latent NDI interpolation inference script.

Loads the trained LatentNDIModel (exp 18) and generates hourly NetCDF outputs
from 6-hourly MEPS data for 2024 (first 10 days of each month).

Unlike the SLERP-based _interpolator.py, intermediate frames are predicted by
the learned LatentFCInterpolator network rather than geometric interpolation.

Ensemble variety is produced by sampling N_ENSEMBLE times from the VAE posterior
of each boundary frame and running the deterministic network per sample.

Output path: _saved/output-bis-2nd-report-00/
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import zarr
from pathlib import Path
from tqdm import tqdm

from dssml.data.helpers.normalizers import NormalizerBase
from dssml.data.helpers.helpers import resolve_var_indices

LOGGER = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "/mnt/tier2/project/p200177/u101329/DE371_bis/diffusion-interpolation/_assets/18-lnd-z32-qrl-vae-20260612_174451/last.ckpt"
DATASET_PATH    = "/mnt/tier1/project/p200177/DE_371_bis/meps-2p5km-2020-2025-1h-v2_subdomain_subvars_rechu.zarr"
OUTPUT_DIR      = "/mnt/tier2/project/p200177/u101329/DE371_bis/diffusion-interpolation/_saved/output-bis-2nd-report-00"
LAT_LON_SOURCE  = "/project/home/p200177/DE_371/metno_interpolator/data/interpolator_MEPS_2024-01_11_memb.nc"

N_ENSEMBLE      = 11
STRIDE          = 6          # boundary spacing: t0→t6, t6→t12, ...
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YEAR            = 2024
DAYS_PER_MONTH  = 10

# 19 variables used by the QRL VAE (zarr store has 23; z/10fg/fog/vis excluded)
VAE_VARIABLES = [
    '10si', '10u', '10v', '2d', '2t', 'cbh', 'hcc', 'lcc', 'lsm', 'mcc',
    'sp', 't_850', 'tcc', 'tcw', 'tp', 'u_850', 'v_850', 'w_850', 'q_850',
]

OUTPUT_VARIABLES = ['10u', '10v', '2t', 'msl', 'tp', 'ws10']

VAR_IDX = {v: VAE_VARIABLES.index(v) for v in VAE_VARIABLES}

RESHAPE = [256, 256]


# ── Minimal normalizer stub ────────────────────────────────────────────────────

class _NormStub(NormalizerBase):
    """
    Placeholder normalizer whose buffers are overwritten by the checkpoint
    state_dict when load_from_checkpoint restores the model weights.
    Must match the buffer names of MultiNormalizer (alpha, beta, loss_weights).
    """

    def __init__(self, n_vars: int):
        super().__init__(required_stats=[], var_dim=-4, stats={})
        self.register_buffer("alpha",        torch.ones(n_vars))
        self.register_buffer("beta",         torch.zeros(n_vars))
        self.register_buffer("loss_weights", torch.zeros(n_vars))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        a = self._broadcast("alpha", x)
        b = self._broadcast("beta",  x)
        return (x - b) / a

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        a = self._broadcast("alpha", x)
        b = self._broadcast("beta",  x)
        return x * a + b


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    from dssml.models.interp import LatentNDIModel

    normalizer = _NormStub(len(VAE_VARIABLES))
    model = LatentNDIModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        normalizer=normalizer,
    )
    model.eval()
    model.to(device)
    return model


def read_single_frame(root_data, idx: int, var_indices, reshape):
    """Read one frame from the zarr store. Returns (1, V, E, H, W) tensor."""
    idx = int(idx)
    x = root_data[idx : idx + 1, var_indices, :, :]   # (1, V, E, cell)
    x = torch.from_numpy(x.astype(np.float32))
    lead = x.shape[:-1]
    return x.view(*lead, *reshape)                      # (1, V, E, H, W)


def normalise_frame(model, x_raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    x_raw : (1, V, E, H, W) raw single frame
    returns: (1, V*E, H, W) normalised on device

    MultiNormalizer requires var_dim=-4, which resolves to dim 2 in a 6-D tensor.
    We add a time dimension (unsqueeze(1)) to form (1,1,V,E,H,W) before normalizing.
    """
    x_6d = x_raw.unsqueeze(1).to(device)   # (1, 1, V, E, H, W)
    if model.normalizer is not None:
        x_norm_6d = model.normalizer(x_6d)
    else:
        x_norm_6d = x_6d
    V = x_raw.shape[1]
    E = x_raw.shape[2]
    return x_norm_6d.squeeze(1).view(1, V * E, *x_raw.shape[3:])   # (1, V*E, H, W)


def encode_and_sample(model, x_norm: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    x_norm  : (1, V*E, H, W) normalised single frame
    returns : (n_samples, C_z, H_z, W_z) posterior samples
    """
    ae = model.first_stage.auto_encoder
    ae.eval()
    with torch.no_grad():
        posterior = ae.encode_posterior(x_norm)
        samples = torch.stack(
            [posterior.sample().squeeze(0) for _ in range(n_samples)], dim=0
        )
    return samples  # (n_samples, C_z, H_z, W_z)


def decode_and_denorm(model, z: torch.Tensor, V: int, E: int = 1) -> torch.Tensor:
    """
    z       : (N, C_z, H_z, W_z)
    returns : (N, V, H, W) in physical units  (assumes E=1 in zarr store)
    """
    ae = model.first_stage.auto_encoder
    ae.eval()
    with torch.no_grad():
        x_norm = ae.decode(z)                  # (N, V*E, H, W)
    N, _, H, W = x_norm.shape
    # Reshape to 6-D so MultiNormalizer broadcasts along dim 2 (var_dim=-4)
    x_6d = x_norm.view(N, 1, V, E, H, W)
    if model.normalizer is not None:
        x_phys_6d = model.normalizer.denormalize(x_6d)
    else:
        x_phys_6d = x_6d
    return x_phys_6d.view(N, V, H, W)         # (N, V, H, W)


def extract_output(x_phys: torch.Tensor) -> dict:
    """x_phys : (N_ENSEMBLE, V, H, W) → {var_name: (N_ENSEMBLE, H, W) numpy}."""
    result = {}
    for var_name in OUTPUT_VARIABLES:
        if var_name == "ws10":
            u = x_phys[:, VAR_IDX["10u"], :, :].cpu().numpy()
            v = x_phys[:, VAR_IDX["10v"], :, :].cpu().numpy()
            result["ws10"] = np.sqrt(u**2 + v**2)
        elif var_name == "msl":
            result["msl"] = x_phys[:, VAR_IDX["sp"], :, :].cpu().numpy()
        else:
            result[var_name] = x_phys[:, VAR_IDX[var_name], :, :].cpu().numpy()
    return result


def store_frame(out: dict, frame_idx: int, vals: dict) -> None:
    for v in OUTPUT_VARIABLES:
        out[v][frame_idx] = vals[v]


def get_month_indices(dates: np.ndarray, year: int, month: int, n_days: int):
    """
    Return (start_idx, end_idx) covering [year-month-01, year-month-(n_days+1)] or None.
    """
    start = np.datetime64(f"{year}-{month:02d}-01", "s")
    end   = np.datetime64(f"{year}-{month:02d}-{n_days + 1:02d}", "s")
    mask  = (dates >= start) & (dates <= end)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    return int(indices[0]), int(indices[-1]) + 1


def process_time_range(
    model, root_data, dates, idx_start, idx_end,
    var_indices, reshape, device,
):
    """
    Generate hourly frames for [idx_start, idx_end) using the LND interpolation model.

    For each 6-hour segment (i, i+6):
      - Sample N_ENSEMBLE latent vectors from each boundary posterior
      - Run the interpolation network (N_ENSEMBLE batched) → 5 intermediate latents
      - Decode all frames to physical space
    """
    H, W = reshape
    V    = len(VAE_VARIABLES)
    E    = 1                     # zarr data has E=1
    C_z  = model.latent_channels
    n_interp = model.n_interp   # 5

    n_times = idx_end - idx_start
    ts = dates[idx_start:idx_end]

    # Pre-allocate output arrays
    out = {
        v: np.full((n_times, N_ENSEMBLE, H, W), np.nan, dtype=np.float32)
        for v in OUTPUT_VARIABLES
    }

    i = 0
    n_segments = (n_times - 1) // STRIDE
    pbar = tqdm(total=n_segments, desc="  Segments", leave=False)

    while i + STRIDE <= n_times - 1:
        g0 = idx_start + i
        g6 = idx_start + i + STRIDE

        # Read & normalise boundary frames
        x0_raw  = read_single_frame(root_data, g0, var_indices, reshape)
        x6_raw  = read_single_frame(root_data, g6, var_indices, reshape)
        x0_norm = normalise_frame(model, x0_raw, device)   # (1, V, H, W)
        x6_norm = normalise_frame(model, x6_raw, device)

        # Sample N_ENSEMBLE latents from each posterior
        z0 = encode_and_sample(model, x0_norm, N_ENSEMBLE)  # (N, C_z, H_z, W_z)
        z6 = encode_and_sample(model, x6_norm, N_ENSEMBLE)

        # Predict 5 intermediate latents for all ensemble members in one forward pass
        z_cond = torch.cat([z0, z6], dim=1)                 # (N, 2*C_z, H_z, W_z)
        with torch.no_grad():
            z_pred = model.network(z_cond)                  # (N, n_interp*C_z, H_z, W_z)

        # Decode & store boundary t0
        x_phys_0 = decode_and_denorm(model, z0, V, E)       # (N, V, H, W)
        store_frame(out, i, extract_output(x_phys_0))

        # Decode & store intermediate frames t1 … t5
        for k in range(n_interp):
            z_k  = z_pred[:, k * C_z : (k + 1) * C_z]     # (N, C_z, H_z, W_z)
            x_k  = decode_and_denorm(model, z_k, V, E)
            store_frame(out, i + 1 + k, extract_output(x_k))

        i += STRIDE
        pbar.update(1)

    # Decode & store the final boundary frame
    if i < n_times:
        g_last   = idx_start + i
        x_l_raw  = read_single_frame(root_data, g_last, var_indices, reshape)
        x_l_norm = normalise_frame(model, x_l_raw, device)
        z_last   = encode_and_sample(model, x_l_norm, N_ENSEMBLE)
        x_phys_l = decode_and_denorm(model, z_last, V, E)
        store_frame(out, i, extract_output(x_phys_l))

    pbar.close()

    # Trim to valid frames
    last_valid = min(i + 1, n_times)
    for v in out:
        out[v] = out[v][:last_valid]
    ts = ts[:last_valid]

    return out, ts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)

    LOGGER.info("Loading model from %s", CHECKPOINT_PATH)
    model = load_model(CHECKPOINT_PATH, DEVICE)
    LOGGER.info("  latent_channels=%d  n_interp=%d  device=%s",
                model.latent_channels, model.n_interp, DEVICE)

    LOGGER.info("Opening dataset %s", DATASET_PATH)
    root       = zarr.open(DATASET_PATH, mode="r")
    root_data  = root["data"]
    all_vars   = list(root.attrs.get("variables", []))
    var_indices = resolve_var_indices(VAE_VARIABLES, all_vars)
    dates      = root["dates"][:].astype("datetime64[s]")

    with xr.open_dataset(LAT_LON_SOURCE) as ds_ref:
        lat_data = ds_ref["lat"].values
        lon_data = ds_ref["lon"].values

    H, W = RESHAPE
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for month in range(1, 13):
        LOGGER.info("Processing %d-%02d (first %d days)…", YEAR, month, DAYS_PER_MONTH)

        result = get_month_indices(dates, YEAR, month, DAYS_PER_MONTH)
        if result is None:
            LOGGER.warning("  No data for %d-%02d, skipping.", YEAR, month)
            continue

        idx_start, idx_end = result
        LOGGER.info("  Index range: [%d, %d)  (%d time steps)",
                    idx_start, idx_end, idx_end - idx_start)

        out, ts = process_time_range(
            model, root_data, dates,
            idx_start, idx_end,
            var_indices, RESHAPE, DEVICE,
        )

        # Build NetCDF
        n_out  = len(ts)
        ts_i64 = ts.astype("datetime64[s]").astype(np.int64)

        coords = {
            "time":     ("time",     ts_i64),
            "ensemble": ("ensemble", np.arange(N_ENSEMBLE, dtype=np.int64)),
            "y":        ("y",        np.arange(H, dtype=np.int64)),
            "x":        ("x",        np.arange(W, dtype=np.int64)),
        }

        data_vars = {}
        for var_name in OUTPUT_VARIABLES:
            data_vars[var_name] = xr.Variable(
                dims=["time", "ensemble", "y", "x"],
                data=out[var_name],
                attrs={"_FillValue": np.nan},
            )
        data_vars["lat"] = xr.Variable(dims=["y", "x"], data=lat_data,
                                        attrs={"_FillValue": np.nan})
        data_vars["lon"] = xr.Variable(dims=["y", "x"], data=lon_data,
                                        attrs={"_FillValue": np.nan})

        ds_out   = xr.Dataset(data_vars=data_vars, coords=coords)
        out_path = output_dir / f"interpolator_MEPS_{YEAR}-{month:02d}_11_memb.nc"
        ds_out.to_netcdf(str(out_path), mode="w")
        LOGGER.info("  Wrote %s", out_path)

    LOGGER.info("All months done.")


if __name__ == "__main__":
    main()
