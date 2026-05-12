# inference.py

import numpy as np
import nibabel as nib
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import List, Tuple, Optional
import os
import json
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import csv

from src.optimization.utils import get_slice_params
from src.optimization.renderer import simulate_slice_pixels, simulate_slice_pixels_hetero



# --------------------------------------------------------------------------
# Slice-wise evaluation and visualization
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full_stack(
    inr,
    slice_mod,
    img_nib,
    img_data,
    metas,
    rot_mode,
    style,
    N_ax,
    N_co,
    slice_axis: int,
    normalizer,
    stack_id: int,
    K: int = 64,
    device: str = "cpu",
    normalize_sigma: bool = True,
):
    """
    Renders every slice in a stack (no saving) and returns mean MAE/PSNR/SSIM.
    Uses the same rendering path as inference (_render_one_slice).
    """

    if stack_id == 0:
        slices_ids = (0, len(metas) - 1)
    else:
        slices_ids = (0, len(metas) - 1)
    maes, psnrs, ssims, nccs = [], [], [], []

    print(f"stack_id={stack_id} slices={slices_ids}")
    for s in range(slices_ids[0], slices_ids[1] + 1):
        pred, gt = _render_one_slice(
            inr=inr,
            slice_mod=slice_mod,
            img_nib=img_nib,
            img_data=img_data,
            metas=metas,
            slice_axis=slice_axis,
            s=s,
            N_ax=N_ax,
            N_co=N_co,
            normalizer=normalizer,
            stack_id=stack_id,
            K=K,
            device=device,
            normalize_sigma=normalize_sigma,
            rot_mode=rot_mode,
            style=style,
        )


        gt_v = gt
        pred_v = pred

        maes.append(_mae(pred_v, gt_v))
        psnrs.append(_psnr(pred_v, gt_v))
        ssims.append(_ssim(pred_v, gt_v))
        nccs.append(_ncc(pred_v, gt_v))

    # Means for this stack
    return {
        "n": slices_ids[1] - slices_ids[0] + 1,
        "MAE": float(np.mean(maes)),
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "NCC": float(np.mean(nccs)),
    }


def _append_metrics_csv(csv_path: str, row: dict):
    """
    Append metrics to a CSV (creates header if file doesn't exist).
    """
    fieldnames = ["tag", "slice_index", "it", "K", "MAE", "PSNR", "SSIM", "png_path"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


@torch.no_grad()
def _render_one_slice(
    inr,
    slice_mod,
    img_nib,
    img_data,
    metas,
    slice_axis,
    s,
    normalizer,
    N_ax,
    N_co,
    rot_mode,
    style,
    stack_id: int,
    K: int = 64,
    device: str = "cpu",
    chunk: int = 65536,
    normalize_sigma: bool = True,
):
    """
    Render a full slice s of a given stack (axial or coronal).

    Returns:
      pred_slice: (H,W) float32
      gt_slice  : (H,W) float32
    """
    vol = img_data

    shape = vol.shape
    in_axes = [ax for ax in (0, 1, 2) if ax != slice_axis]
    W = shape[in_axes[0]]
    H = shape[in_axes[1]]

    # per-pixel (i,j,s)
    ii, jj = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy",
    )
    ij = np.stack([ii.reshape(-1), jj.reshape(-1)], axis=-1)  # (HW,2)
    s_idx = np.full((ij.shape[0],), s, dtype=np.int64)  # (HW,)

    # slice center for this s, broadcast to pixels
    ctr = np.asarray(metas[s].center_world, dtype=np.float32)
    ctr = np.broadcast_to(ctr, (ij.shape[0], 3)).copy()

    # --- compute pose ONCE per slice, then broadcast ---
    sid1 = torch.tensor([int(stack_id)], dtype=torch.long, device=device)
    idx1 = torch.tensor([int(s)], dtype=torch.long, device=device)
    # The slice_mod now returns w, t, sigma_logit, and scale_logit
    pose1 = slice_mod(sid1, idx1) 
    pose = {k: v.expand(ij.shape[0], *v.shape[1:]) for k, v in pose1.items()}

    # --- Start of Modified Parameter Retrieval ---
    if normalize_sigma:
        with torch.no_grad():
            # MODIFICATION 1: get_slice_params now returns four values: sig, scale, w, t
            scale_all, w_all, t_all = get_slice_params(
                slice_mod, N_ax, N_co, device
            )

            # MODIFICATION 2: Calculate index and retrieve scale and sigma separately
            idx = s if stack_id == 0 else (N_ax + s)
            
            # Intensity Scale (C_i)
            scale_one = scale_all[idx].view(1, 1).to(device)
            scale_vals = scale_one.expand(ij.shape[0], 1)

    else:
        # Default behavior if not normalizing: scale is 1, sigma is a small positive value
        scale_vals = torch.ones(
            (ij.shape[0], 1), dtype=torch.float32, device=device
        )



    # render in chunks to keep VRAM in check
    pred = np.zeros((ij.shape[0],), dtype=np.float32)
    start = 0
    while start < ij.shape[0]:
        end = min(start + chunk, ij.shape[0])
        if style == "hetero":
            out = simulate_slice_pixels_hetero(
                inr=inr,
                affine_np=img_nib.affine,
                zooms_xyz=img_nib.header.get_zooms()[:3],
                slice_axis=slice_axis,
                slice_indices=torch.as_tensor(
                    s_idx[start:end], dtype=torch.long, device=device
                ),
                ij_pixels=torch.from_numpy(ij[start:end]).to(device),
                normalizer=normalizer,
                # MODIFICATION 3: Pass both decoupled parameters
                pose_params={
                    "w": pose["w"][start:end],
                    "t": pose["t"][start:end],
                    "scale": scale_vals[start:end],  # New: Intensity Scale
                    # Removed "alpha" parameter
                },
                slice_centers_world=torch.from_numpy(ctr[start:end].copy()).to(
                    device
                ),
                K=K,
                device=device,
                rot_mode=rot_mode,
            )
            pred_chunk = out["mu"]
        else:
            # The homogeneous case (simulate_slice_pixels) still uses "alpha"
            # Assuming you want to keep the homogeneous path for backward compatibility,
            # we will pass the scale parameter as "alpha" (the old intensity-affecting parameter).
            pred_chunk = simulate_slice_pixels(
                inr=inr,
                affine_np=img_nib.affine,
                zooms_xyz=img_nib.header.get_zooms()[:3],
                slice_axis=slice_axis,
                slice_indices=torch.as_tensor(
                    s_idx[start:end], dtype=torch.long, device=device
                ),
                ij_pixels=torch.from_numpy(ij[start:end]).to(device),
                normalizer=normalizer,
                pose_params={
                    "w": pose["w"][start:end],
                    "t": pose["t"][start:end],
                    "scale": scale_vals[start:end],  # Pass scale as alpha for homogeneous case
                },
                slice_centers_world=torch.from_numpy(ctr[start:end].copy()).to(
                    device
                ),
                K=K,
                device=device,
                rot_mode=rot_mode,
            )

        pred[start:end] = pred_chunk.detach().cpu().numpy()
        start = end

    # shape to image (H,W)
    pred_img = pred.reshape(H, W)

    # ground-truth slice in same (H,W) layout
    gt = vol.take(indices=s, axis=slice_axis).astype(np.float32)  # (W,H)
    gt = gt.T  # (H,W)
    return pred_img, gt


def _psnr(a: np.ndarray, b: np.ndarray, data_range=1.0):
    return float(peak_signal_noise_ratio(b, a, data_range=data_range))


def _ssim(a: np.ndarray, b: np.ndarray, data_range=1.0):
    return float(structural_similarity(b, a, data_range=data_range, channel_axis=None))


def _mae(a: np.ndarray, b: np.ndarray):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _ncc(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (a0.std() * b0.std()) + eps
    return float((a0 * b0).mean() / denom)


@torch.no_grad()
def infer_and_save_slice(
    *,
    inr,
    slice_mod,
    img_nib,
    img_data,
    metas,
    rot_mode,
    style,
    slice_axis: int,
    slice_index: int,
    normalizer,
    N_ax,
    N_co,
    stack_id: int,  # 0 = axial, 1 = coronal
    run_dirs: dict,  # {"stamp":..., "run_dir":..., "axial":..., "coronal":...}
    K: int = 64,
    device: str = "cpu",
    normalize_sigma: bool = True,
    dpi: int = 140,
    step: Optional[int] = None,  # logged into filename if provided
    wandb_log: bool = True,
):
    """
    Renders one slice and saves:
      - PNG into <run>/axial/ or <run>/coronal/
      - appends/creates <run>/metrics.csv
    """
    # which subfolder + prefix
    tag = "axial" if stack_id == 0 else "coronal"
    subdir = run_dirs["axial"] if stack_id == 0 else run_dirs["coronal"]
    os.makedirs(subdir, exist_ok=True)
    prefix = tag

    # render (pred, gt already normalized to [0,1] if eval_normalize_gt=True inside)
    pred, gt = _render_one_slice(
        inr=inr,
        slice_mod=slice_mod,
        img_nib=img_nib,
        img_data=img_data,
        metas=metas,
        slice_axis=slice_axis,
        s=slice_index,
        normalizer=normalizer,
        stack_id=stack_id,
        K=K,
        device=device,
        N_ax=N_ax,
        N_co=N_co,
        normalize_sigma=normalize_sigma,
        rot_mode=rot_mode,
        style=style,
    )


    gt_v = gt
    pred_v = pred


    # metrics
    mae_raw = _mae(pred_v, gt_v)
    psnr_raw = _psnr(pred_v, gt_v)
    ssim_raw = _ssim(pred_v, gt_v)
    ncc_raw = _ncc(pred_v, gt_v)

    # filename
    fname = (
        f"{prefix}_s{slice_index:03d}.png"
        if step is None
        else f"{prefix}_it{step:06d}_s{slice_index:03d}.png"
    )
    out_path = os.path.join(subdir, fname)

    # plot & save
    fig, axs = plt.subplots(2, 1, figsize=(4, 8), dpi=dpi)
    axs[0].imshow(gt_v, cmap="gray", origin="lower")
    axs[0].set_title(f"GT ({prefix}, s={slice_index})")
    axs[0].axis("off")
    axs[1].imshow(pred_v, cmap="gray", origin="lower")
    axs[1].set_title(
        f"Pred \nMAE={mae_raw:.3f} PSNR={psnr_raw:.1f} SSIM={ssim_raw:.2f} NCC={ncc_raw:.2f}"
    )
    axs[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # CSV log (one file per run)
    csv_path = os.path.join(run_dirs["run_dir"], "metrics.csv")
    _append_metrics_csv(
        csv_path,
        {
            "tag": tag,
            "slice_index": slice_index,
            "it": step if step is not None else "",
            "K": K,
            "MAE": mae_raw,
            "PSNR": psnr_raw,
            "SSIM": ssim_raw,
            "NCC": ncc_raw,
            "png_path": out_path,
        },
    )

    # W&B log
    if wandb_log:
        try:
            wandb.log(
                {
                    f"{prefix}/MAE_raw": mae_raw,
                    f"{prefix}/PSNR_raw": psnr_raw,
                    f"{prefix}/SSIM_raw": ssim_raw,
                    f"{prefix}/NCC_raw": ncc_raw,
                },
                step=step,
            )
        except Exception:
            pass

    #print("\nSaving...")
    #print(
    #    f"RAW:    MAE={mae_raw:.3f} PSNR={psnr_raw:.1f} SSIM={ssim_raw:.2f} NCC={ncc_raw:.2f}\n"
    #)
    return mae_raw, psnr_raw, ssim_raw, ncc_raw

