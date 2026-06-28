import logging
import numpy as np
import torch
import xarray as xr
import zarr
from pathlib import Path
from tqdm import tqdm

from dssml.networks.ae.ldm_vae import DiagonalGaussian
from dssml.data.helpers.normalizers import SymRangeNormalizer
from dssml.data.helpers.helpers import resolve_var_indices

LOGGER = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "/mnt/tier2/project/p200177/u101329/DE371_bis/diffusion-interpolation/_work/ldm-vae/mse_kl_04_0_20260330_034815/epoch=067-step=111384-val_loss=0.0036.ckpt"

# For the answer 
CHECKPOINT_PATH = "/mnt/tier2/project/p200177/u101329/DE371_bis/diffusion-interpolation/_work/ldm-vae-answer/mse_kl_04_0_20260417_092115/epoch=216-step=355446-val_loss=0.0022.ckpt"


DATASET_PATH    = "/mnt/tier1/project/p200177/DE_371_bis/meps-2p5km-2020-2025-1h-v2_subdomain_subvars_rechu.zarr"
OUTPUT_DIR      = "/mnt/tier2/project/p200177/u101329/DE371_bis/diffusion-interpolation/_saved/outputs_answer"
LAT_LON_SOURCE  = "/project/home/p200177/DE_371/metno_interpolator/data/interpolator_MEPS_2024-01_11_memb.nc"

N_ENSEMBLE      = 11
STRIDE          = 6          # boundary spacing: t0→t6, t6→t12, ...
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YEAR            = 2024
DAYS_PER_MONTH  = 10

VAE_VARIABLES = [
    '10si','10u','10v','2d','2t','cbh','hcc','lcc','lsm','mcc',
    'sp','t_850','tcc','tcw','tp','u_850','v_850','w_850','q_850','z'
]

OUTPUT_VARIABLES = ['10u', '10v', '2t', 'msl', 'tp', 'ws10']

VAR_IDX = {v: VAE_VARIABLES.index(v) for v in VAE_VARIABLES}

RESHAPE = [256, 256]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    from dssml.models.ae.ldm_vae import LDMVAEModel

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"]

    normalizer = None
    if "normalizer.minimum" in state_dict:
        stats = {
            "minimum": state_dict["normalizer.minimum"],
            "maximum": state_dict["normalizer.maximum"],
        }
        norm_const = state_dict.get(
            "normalizer.norm_const_tensor", torch.tensor(1.0)
        ).item()
        normalizer = SymRangeNormalizer(stats=stats, norm_const=norm_const)

    model = LDMVAEModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        normalizer=normalizer,
    )
    model.eval()
    model.to(device)
    return model


def read_single_frame(root_data, idx, var_indices, reshape):
    """Read one frame from the zarr store. Returns (1, V, E, H, W) tensor."""
    idx = int(idx)
    x = root_data[idx : idx + 1, var_indices, :, :]  # (1, V, E, cell)
    x = torch.from_numpy(x.astype(np.float32))
    lead = x.shape[:-1]
    return x.view(*lead, *reshape)  # (1, V, E, H, W)


def normalise_frame(model, x_raw, device):
    """x_raw: (1, V, E, H, W) → (1, V*E, H, W) normalised on device."""
    _, V, E, H, W = x_raw.shape
    x_flat = x_raw.view(1, V * E, H, W).to(device)
    if model.normalizer is not None:
        x_norm = (
            model.normalizer(x_raw.unsqueeze(0).to(device))  # (1,1,V,E,H,W)
            .squeeze(0)                                       # (1,V,E,H,W)
            .view(1, V * E, H, W)
        )
    else:
        x_norm = x_flat
    return x_norm


def encode_frame(model, frame: torch.Tensor) -> DiagonalGaussian:
    """frame: (1, C, H, W) normalised → DiagonalGaussian posterior."""
    with torch.no_grad():
        return model.auto_encoder.encode(frame)


def decode_and_denorm(model, z: torch.Tensor) -> torch.Tensor:
    """z (N, C_z, H_z, W_z) → decoded → denormalised (N, C, H, W)."""
    with torch.no_grad():
        x_hat = model.auto_encoder.decode(z)
    if model.normalizer is None:
        return x_hat
    n, c, h, w = x_hat.shape
    x6d = x_hat.unsqueeze(1).unsqueeze(3)       # (N, 1, C, 1, H, W)
    x6d = model.normalizer.denormalize(x6d)
    return x6d.squeeze(1).squeeze(2)             # (N, C, H, W)


def sample_posterior(posterior: DiagonalGaussian, n: int) -> torch.Tensor:
    """Draw n samples from a DiagonalGaussian. Returns (n, C_z, H_z, W_z)."""
    return torch.stack(
        [posterior.sample().squeeze(0) for _ in range(n)], dim=0
    )


def extract_output(x_phys: torch.Tensor) -> dict:
    """x_phys: (N_ENSEMBLE, C, H, W) → {var_name: (N_ENSEMBLE, H, W) numpy}."""
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


def store_frame(out, frame_idx, vals):
    """Store extracted values into the output arrays at frame_idx."""
    for v in OUTPUT_VARIABLES:
        out[v][frame_idx] = vals[v]


def slerp(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation. z0, z1: same shape. alpha: 0→z0, 1→z1."""
    z0_flat = z0.reshape(-1).float()
    z1_flat = z1.reshape(-1).float()

    cos_omega = torch.nn.functional.cosine_similarity(
        z0_flat.unsqueeze(0), z1_flat.unsqueeze(0)
    ).clamp(-1, 1)
    omega = torch.acos(cos_omega)

    if omega.abs() < 1e-6:
        return (1.0 - alpha) * z0 + alpha * z1

    sin_omega = torch.sin(omega)
    w0 = torch.sin((1.0 - alpha) * omega) / sin_omega
    w1 = torch.sin(alpha * omega) / sin_omega
    return w0 * z0 + w1 * z1


def get_month_indices(dates: np.ndarray, year: int, month: int, n_days: int):
    """
    Given a datetime64[s] date array, return (start_idx, end_idx) covering
    [year-month-01 00:00, year-month-(n_days+1) 00:00] inclusive, or None.
    """
    start = np.datetime64(f"{year}-{month:02d}-01", "s")
    end = np.datetime64(f"{year}-{month:02d}-{n_days + 1:02d}", "s")
    mask = (dates >= start) & (dates <= end)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    return int(indices[0]), int(indices[-1]) + 1


def encode_single(model, root_data, g_idx, var_indices, reshape, device):
    """Read, normalise, encode one frame. Returns DiagonalGaussian posterior."""
    x_raw = read_single_frame(root_data, g_idx, var_indices, reshape)
    x_norm = normalise_frame(model, x_raw, device)
    return encode_frame(model, x_norm)


def process_time_range(model, root_data, dates, idx_start, idx_end,
                       var_indices, reshape, device):
    """
    Process [idx_start, idx_end):

    i = 0
    while i + STRIDE <= n_times - 1:
        encode frame[i] and frame[i+6] → posteriors p0, p6
        sample 11 latents from each    → z0[0..10], z6[0..10]
        decode z0 samples               → store frame i
        for k in 1..5:
            slerp(z0[ens], z6[ens], k/6) per ensemble member
            decode → store frame i+k
        i += 6
    store last boundary frame
    """
    H, W = reshape
    n_times = idx_end - idx_start
    ts = dates[idx_start:idx_end]  # datetime64[s] slice

    # Pre-allocate output
    out = {
        v: np.full((n_times, N_ENSEMBLE, H, W), np.nan, dtype=np.float32)
        for v in OUTPUT_VARIABLES
    }

    # ── Walk through segments ───────────────────────────────────────────
    i = 0
    n_segments = (n_times - 1) // STRIDE
    pbar = tqdm(total=n_segments, desc="  Segments", leave=False)

    while i + STRIDE <= n_times - 1:
        g0 = idx_start + i
        g6 = idx_start + i + STRIDE

        # Encode both boundaries
        posterior_0 = encode_single(model, root_data, g0, var_indices, reshape, device)
        posterior_6 = encode_single(model, root_data, g6, var_indices, reshape, device)

        # Sample N_ENSEMBLE latents from each
        z0 = sample_posterior(posterior_0, N_ENSEMBLE)  # (N, C_z, H_z, W_z)
        z6 = sample_posterior(posterior_6, N_ENSEMBLE)

        # Decode & store boundary t0
        x_phys_0 = decode_and_denorm(model, z0)
        store_frame(out, i, extract_output(x_phys_0))

        # Interpolate & decode interior frames t1..t5
        for k in range(1, STRIDE):
            alpha = k / STRIDE

            z_interp = torch.stack(
                [slerp(z0[ens], z6[ens], alpha) for ens in range(N_ENSEMBLE)],
                dim=0,
            )
            x_phys = decode_and_denorm(model, z_interp)
            store_frame(out, i + k, extract_output(x_phys))

        i += STRIDE
        pbar.update(1)

    # ── Store the very last boundary ────────────────────────────────────
    if i < n_times:
        posterior_last = encode_single(
            model, root_data, idx_start + i, var_indices, reshape, device
        )
        z_last = sample_posterior(posterior_last, N_ENSEMBLE)
        x_phys_last = decode_and_denorm(model, z_last)
        store_frame(out, i, extract_output(x_phys_last))

    pbar.close()

    # Trim to actual valid frames
    last_valid = min(i + 1, n_times)
    for v in out:
        out[v] = out[v][:last_valid]
    ts = ts[:last_valid]

    return out, ts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)

    # 1. Load model
    LOGGER.info("Loading model...")
    model = load_model(CHECKPOINT_PATH, DEVICE)

    # 2. Open dataset & metadata
    LOGGER.info("Opening dataset...")
    root = zarr.open(DATASET_PATH, mode="r")
    root_data = root["data"]

    # Use resolve_var_indices from helpers (validates names, returns np.int32 array)
    all_vars = list(root.attrs.get("variables", []))
    var_indices = resolve_var_indices(VAE_VARIABLES, all_vars)

    # Load dates as datetime64[s] (same convention as MepsZarrManifest)
    dates = root["dates"][:].astype("datetime64[s]")

    # Read lat/lon once
    with xr.open_dataset(LAT_LON_SOURCE) as ds_ref:
        lat_data = ds_ref["lat"].values
        lon_data = ds_ref["lon"].values

    H, W = RESHAPE
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Loop over each month of 2024
    for month in range(1, 13):
        LOGGER.info(
            f"Processing {YEAR}-{month:02d} (first {DAYS_PER_MONTH} days)..."
        )

        result = get_month_indices(dates, YEAR, month, DAYS_PER_MONTH)
        if result is None:
            LOGGER.warning(f"  No data for {YEAR}-{month:02d}, skipping.")
            continue

        idx_start, idx_end = result
        LOGGER.info(
            f"  Index range: [{idx_start}, {idx_end})  "
            f"({idx_end - idx_start} time steps)"
        )

        out, ts = process_time_range(
            model, root_data, dates,
            idx_start, idx_end,
            var_indices, RESHAPE, DEVICE,
        )

        # ── Build NetCDF ────────────────────────────────────────────────
        n_out = len(ts)
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
        data_vars["lat"] = xr.Variable(
            dims=["y", "x"], data=lat_data, attrs={"_FillValue": np.nan}
        )
        data_vars["lon"] = xr.Variable(
            dims=["y", "x"], data=lon_data, attrs={"_FillValue": np.nan}
        )

        ds_out = xr.Dataset(data_vars=data_vars, coords=coords)
        out_path = output_dir / f"interpolator_MEPS_{YEAR}-{month:02d}_11_memb.nc"
        ds_out.to_netcdf(str(out_path), mode="w")
        LOGGER.info(f"  Wrote {out_path}")

    LOGGER.info("All months done.")


if __name__ == "__main__":
    main()