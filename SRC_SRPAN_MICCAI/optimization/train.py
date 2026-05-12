# train_inr.py

# --- 1. Standard Library ---
import os
import json
import math
import time
from pathlib import Path
from typing import Optional

# --- 2. Third-Party ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import nibabel as nib
from tqdm import tqdm
import wandb
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# --- 3. Local Modules ---
from src.optimization.utils import (
    CoordNormalizer, pose_sigma_stats, p95, p05, 
    _append_results_to_excel, get_slice_params
)
from src.models.siren import Siren, SirenHetero
from src.models.slice_module import SliceModuleSIREN
from src.optimization.renderer import simulate_slice_pixels, simulate_slice_pixels_hetero, pixel_ij_to_world_batched
from src.optimization.inference import infer_and_save_slice, evaluate_full_stack
from src.data_processing.dataset import load_volume, slice_metadata, save_stack_meta_json


# ==============================================================================
#   Helper Functions
# ==============================================================================

def sample_volume_world_grid_sample(vol_t: torch.Tensor, inv_affine_t: torch.Tensor, world_pts: torch.Tensor, return_valid: bool = True):
    """
    Optimized sampling using torch.nn.functional.grid_sample.
    Used for GATING (sampling at arbitrary world coordinates).
    """
    B = world_pts.shape[0]
    device = vol_t.device
    X, Y, Z = vol_t.shape

    # 1. World -> Voxel Coords
    ones = torch.ones((B, 1), dtype=world_pts.dtype, device=device)
    # (B, 4) @ (4, 4).T -> (B, 3)
    ijk = (inv_affine_t @ torch.cat([world_pts, ones], dim=1).T).T[:, :3]
    i, j, k = ijk.unbind(-1)

    # 2. Valid Mask (needed for gating)
    valid = (i >= 0) & (i <= X - 1) & (j >= 0) & (j <= Y - 1) & (k >= 0) & (k <= Z - 1)

    # 3. Normalize to [-1, 1] for grid_sample
    # grid_sample uses (x, y, z) convention which maps to (W, H, D) or (k, j, i)
    norm_k = 2.0 * (k / (Z - 1)) - 1.0
    norm_j = 2.0 * (j / (Y - 1)) - 1.0
    norm_i = 2.0 * (i / (X - 1)) - 1.0

    # 4. Sample
    # Stack order must be (x, y, z) -> (k, j, i) for grid_sample
    grid = torch.stack([norm_k, norm_j, norm_i], dim=-1).view(1, 1, 1, B, 3)
    input_vol = vol_t.view(1, 1, X, Y, Z)
    
    # align_corners=True matches standard voxel indexing
    vals = F.grid_sample(input_vol, grid, mode='bilinear', padding_mode='zeros', align_corners=True).view(-1)

    return (vals, valid) if return_valid else vals


def sample_uniform_gpu(vol_t, slice_axis, n_pixels):
    """
    Samples random pixels directly from the GPU volume tensor.
    Used for TRAINING DATA (integer pixel coordinates).
    Returns indices and values already on the correct device.
    """
    shape = vol_t.shape
    # Identify the dimensions that are NOT the slice axis
    in_axes = [ax for ax in (0, 1, 2) if ax != slice_axis]
    
    # Random indices generated directly on GPU
    idx_slice = torch.randint(0, shape[slice_axis], (n_pixels,), device=vol_t.device)
    idx_i = torch.randint(0, shape[in_axes[0]], (n_pixels,), device=vol_t.device)
    idx_j = torch.randint(0, shape[in_axes[1]], (n_pixels,), device=vol_t.device)

    # Construct 3D indices to gather values
    indices = [None, None, None]
    indices[slice_axis] = idx_slice
    indices[in_axes[0]] = idx_i
    indices[in_axes[1]] = idx_j
    
    # Gather values using advanced indexing
    vals = vol_t[indices[0], indices[1], indices[2]]
    
    # Stack i,j for coordinate mapping (float needed for renderer)
    ij = torch.stack([idx_i.float(), idx_j.float()], dim=-1)
    
    return idx_slice, ij, vals


def make_run_dirs(base_out_dir: str = "results/validation", stamp: Optional[str] = None, run_name: Optional[str] = None):
    if stamp is None:
        stamp = run_name or time.strftime("%Y-%m-%d_%H-%M-%S")
    
    stamp = str(stamp).replace(" ", "_").replace("/", "_")
    run_dir = os.path.join(base_out_dir, stamp)
    
    dirs = {
        "stamp": stamp,
        "run_dir": run_dir,
        "axial": os.path.join(run_dir, "axial"),
        "coronal": os.path.join(run_dir, "coronal")
    }
    for d in [dirs["axial"], dirs["coronal"]]:
        os.makedirs(d, exist_ok=True)
    return dirs


def K_schedule(iter_cur: int, iter_max: int, K_max: int = 64) -> int:
    x = float(iter_cur) / max(1, int(iter_max))
    return max(1, int(round(K_max * (x**2))))


def make_normalizer_from_metas(metas_a, metas_c, device):
    """
    Compute a world-space bounding box covering both stacks.
    """
    pts = []
    for m in metas_a + metas_c:
        pts.extend(m.fov_corners_world)
    P = np.asarray(pts, dtype=np.float32)
    
    world_min, world_max = P.min(0), P.max(0)
    print("CoordNormalizer (Tight Box): world_min =", world_min, "world_max =", world_max)
    return CoordNormalizer(world_min, world_max, device=device)


def joint_robust_normalize(ax_vol: np.ndarray, co_vol: np.ndarray, p_min=1.0, p_max=99.0):
    ax_vals = ax_vol[ax_vol > 1e-3]
    co_vals = co_vol[co_vol > 1e-3]
    
    if len(ax_vals) == 0 or len(co_vals) == 0:
        print("WARNING: Volume seems empty.")
        return ax_vol, co_vol

    pooled = np.concatenate([ax_vals, co_vals])
    vmin, vmax = np.percentile(pooled, [p_min, p_max])
    print(f"Joint Normalization: p{p_min}={vmin:.4f}, p{p_max}={vmax:.4f}")

    div = vmax - vmin + 1e-8
    ax_norm = (ax_vol - vmin) / div
    co_norm = (co_vol - vmin) / div

    print(f"ax norm min max: {ax_norm.min():.4f} {ax_norm.max():.4f}")
    print(f"co norm min max: {co_norm.min():.4f} {co_norm.max():.4f}")

    return (ax_norm.astype(np.float32), co_norm.astype(np.float32))

@torch.no_grad()
def _counts_check(ax_np, ax_slice_axis, co_np, co_slice_axis):
    return ax_np.shape[ax_slice_axis], co_np.shape[co_slice_axis]

# ==============================================================================
#   Training Loop
# ==============================================================================

def train_one_case(
    ax_img_nib, ax_metas, ax_slice_axis,
    co_img_nib, co_metas, co_slice_axis,
    rot_mode, excel_sheet,
    alpha_iter: int = 125,
    batch_size: int = 6000,
    device: str = "cuda",
    run_dirs: Optional[dict] = None,
    K: int = 64,
    test_batch: bool = False,
    hidden=330,
    depth=6,
    omega_siren=30,
    save_to_excel: bool = False,
    excel_path: str = "results/metrics.xlsx",
    run_name_for_excel: Optional[str] = None,
    style='hetero',
    gating=True
):
    device = torch.device(device)
    print(f"DEVICE: {device}")

    # --- 1. Init Models ---
    if style == 'hetero':
        inr = SirenHetero(hidden=hidden, depth=depth, w0_hidden=omega_siren).to(device)
    else:
        inr = Siren(hidden=hidden, depth=depth, w0_hidden=omega_siren).to(device)

    slice_mod = SliceModuleSIREN(
        n_slices_per_stack=[len(ax_metas), len(co_metas)],
        hidden=64, device=device, rot_mode=rot_mode
    ).to(device)

    # --- 2. Normalization ---
    normalizer = make_normalizer_from_metas(ax_metas, co_metas, device)
    
    ax_raw = ax_img_nib.get_fdata().astype(np.float32)
    co_raw = co_img_nib.get_fdata().astype(np.float32)
    
    # Joint Robust Normalization
    ax_np, co_np = joint_robust_normalize(ax_raw, co_raw, p_min=0.5, p_max=99.5)

    # Save debug volumes
    nib.save(nib.Nifti1Image(ax_np, ax_img_nib.affine), os.path.join(run_dirs['run_dir'], 'axial_preprocessed.nii.gz'))
    nib.save(nib.Nifti1Image(co_np, co_img_nib.affine), os.path.join(run_dirs['run_dir'], 'coronal_preprocessed.nii.gz'))
    print(f"Saved preprocessed stacks to {run_dirs['run_dir']}")

    N_ax, N_co = _counts_check(ax_np, ax_slice_axis, co_np, co_slice_axis)

    ax_vol_t = torch.as_tensor(ax_np, dtype=torch.float32, device=device).contiguous()
    co_vol_t = torch.as_tensor(co_np, dtype=torch.float32, device=device).contiguous()
    
    inv_ax_aff_t = torch.linalg.inv(torch.as_tensor(ax_img_nib.affine, dtype=torch.float32, device=device))
    co_aff_t = torch.as_tensor(co_img_nib.affine, dtype=torch.float32, device=device)

    total_pixels = int(np.prod(ax_img_nib.shape) + np.prod(co_img_nib.shape))
    iters_max = int(np.ceil(alpha_iter * (total_pixels / float(batch_size))))

    opt_inr = Adam(inr.parameters(), lr=5e-5)
    opt_slice = Adam(slice_mod.parameters(), lr=5e-4)
    sch_inr = CosineAnnealingLR(opt_inr, T_max=iters_max, eta_min=2.5e-5)
    sch_slice = CosineAnnealingLR(opt_slice, T_max=iters_max, eta_min=2.5e-5)

    print("Starting training...")
    with tqdm(total=iters_max, desc="Training", ncols=100) as pbar:
        progressed = 0
        for it in range(iters_max):
            inr.train(); slice_mod.train()
            K_iter = K_schedule(it, iters_max, K_max=K)
            
            n_ax = int(batch_size * 0.5)
            n_co = int(batch_size - n_ax)

            # --- SAMPLING (Optimized: GPU Uniform) ---
            # sample_uniform_gpu returns tensors ALREADY on device
            s_idx_ax, ij_ax, y_ax_t = sample_uniform_gpu(ax_vol_t, ax_slice_axis, n_ax)
            s_idx_co, ij_co, y_co_t = sample_uniform_gpu(co_vol_t, co_slice_axis, n_co)

            # Get Params
            scale_all, w_all, t_all = get_slice_params(slice_mod, N_ax, N_co, device)
            
            ctr_ax = torch.tensor([ax_metas[int(si)].center_world for si in s_idx_ax], dtype=torch.float32, device=device)
            ctr_co = torch.tensor([co_metas[int(si)].center_world for si in s_idx_co], dtype=torch.float32, device=device)

            # Gather params helper
            scale_all_ax = scale_all[s_idx_ax].unsqueeze(-1)
            w_ax_b = w_all[s_idx_ax]
            t_ax_b = t_all[s_idx_ax]

            scale_all_co = scale_all[N_ax + s_idx_co].unsqueeze(-1)
            w_co_b = w_all[N_ax + s_idx_co]
            t_co_b = t_all[N_ax + s_idx_co]

            # --- Forward ---
            if style == 'hetero':
                out_ax = simulate_slice_pixels_hetero(
                    inr=inr, affine_np=ax_img_nib.affine, zooms_xyz=ax_img_nib.header.get_zooms()[:3],
                    slice_axis=ax_slice_axis, slice_indices=s_idx_ax, ij_pixels=ij_ax,
                    normalizer=normalizer,
                    pose_params={"w": w_ax_b, "t": t_ax_b, "scale": scale_all_ax},
                    slice_centers_world=ctr_ax, K=K_iter, device=str(device), rot_mode=rot_mode
                )

                out_co = simulate_slice_pixels_hetero(
                    inr=inr, affine_np=co_img_nib.affine, zooms_xyz=co_img_nib.header.get_zooms()[:3],
                    slice_axis=co_slice_axis, slice_indices=s_idx_co, ij_pixels=ij_co,
                    normalizer=normalizer,
                    pose_params={"w": w_co_b, "t": t_co_b, "scale": scale_all_co},
                    slice_centers_world=ctr_co, K=K_iter, device=str(device), rot_mode=rot_mode
                )
                
                mu_ax, logv_ax = out_ax["mu"], out_ax["logvar"]
                mu_co, logv_co = out_co["mu"], out_co["logvar"]

                logv_max = math.log(0.35**2)
                logv_ax = torch.clamp(logv_ax, max=logv_max)
                logv_co = torch.clamp(logv_co, max=logv_max)

                r_ax = mu_ax - y_ax_t
                r_co = mu_co - y_co_t

                # Axial NLL
                loss_ax = (0.5 * (torch.exp(-logv_ax) * r_ax.square() + logv_ax)).mean()

                if gating:
                    logv_c = logv_co
                    data_term = torch.exp(-logv_c) * r_co.square()
                    var_term = logv_c

                    # Gating Check (Uses grid_sample for arbitrary world coords)
                    world_pts = pixel_ij_to_world_batched(co_aff_t, co_slice_axis, s_idx_co, ij_co)
                    with torch.no_grad():
                        y_ax_w, valid = sample_volume_world_grid_sample(ax_vol_t, inv_ax_aff_t, world_pts)
                        
                        diff = (y_ax_w - y_co_t).abs()
                        TAU_MIN = 0.05
                        valid_diff = diff[valid]
                        tau_b = torch.quantile(valid_diff, 0.90) if valid_diff.numel() > 0 else torch.tensor(TAU_MIN, device=device)
                        TAU = torch.maximum(tau_b, torch.tensor(TAU_MIN, device=device))
                        
                        t_prog = it / max(1, iters_max)
                        W_MIN = (1.0 - t_prog) * 0.15 + t_prog * 0.0
                        w_down = W_MIN + (1.0 - W_MIN) / (1.0 + (diff / (TAU + 1e-6)).pow(3.0))
                        w_down = w_down * valid.float()

                    w_sum = w_down.sum() + 1e-8
                    loss_co = (0.5 * (w_down * (data_term + 0.5 * var_term)).sum()) / w_sum
                else:
                    loss_co = (0.5 * (torch.exp(-logv_co) * r_co.square() + logv_co)).mean()
                    w_down, TAU = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            
            else:
                # Standard L1
                pred_ax = simulate_slice_pixels(
                    inr=inr, affine_np=ax_img_nib.affine, zooms_xyz=ax_img_nib.header.get_zooms()[:3],
                    slice_axis=ax_slice_axis, slice_indices=s_idx_ax, ij_pixels=ij_ax,
                    normalizer=normalizer,
                    pose_params={"w": w_ax_b, "t": t_ax_b, "scale": scale_all_ax},
                    slice_centers_world=ctr_ax, K=K_iter, device=str(device), rot_mode=rot_mode
                )
                pred_co = simulate_slice_pixels(
                    inr=inr, affine_np=co_img_nib.affine, zooms_xyz=co_img_nib.header.get_zooms()[:3],
                    slice_axis=co_slice_axis, slice_indices=s_idx_co, ij_pixels=ij_co,
                    normalizer=normalizer,
                    pose_params={"w": w_co_b, "t": t_co_b, "scale": scale_all_co},
                    slice_centers_world=ctr_co, K=K_iter, device=str(device), rot_mode=rot_mode
                )
                loss_ax = (pred_ax - y_ax_t).abs().mean()
                loss_co = (pred_co - y_co_t).abs().mean()
                w_down, TAU = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            loss = loss_ax + loss_co

            # --- E. Backprop ---
            opt_inr.zero_grad(); opt_slice.zero_grad()
            loss.backward()
            opt_slice.step(); opt_inr.step()
            sch_slice.step(); sch_inr.step()

            # --- F. Logging ---
            if (it % 100 == 0) or (it == iters_max-1):
                deg_all = torch.linalg.norm(w_all.detach(), dim=-1) * (180.0/np.pi)
                tmag_all = torch.linalg.norm(t_all.detach(), dim=-1)
                
                log_payload = {
                    "train/loss_total": loss.item(), 
                    "train/loss_ax": loss_ax.item(),
                    "train/loss_co": loss_co.item(),
                    "train/K": int(K_iter),
                    "gate_down/mean_weight": w_down.mean().item() if torch.is_tensor(w_down) else 0,
                    "gate_down/TAU": TAU.item() if torch.is_tensor(TAU) else 0,
                    "rotation/all_mean": deg_all.mean().item(),
                    "Translation/all_mean": tmag_all.mean().item(),
                }
                try:
                    wandb.log({k: v for k, v in log_payload.items() if v is not None})
                except Exception:
                    pass
                
                inc = (it + 1) - progressed
                pbar.update(inc)
                progressed = it + 1

            # Occasional visuals (Restored from Old Code)
            if (it % 100) == 0 or it == (iters_max - 1):
                inr.eval(); slice_mod.eval()
                ax_idx = np.random.randint(0, len(ax_metas))
                co_idx = np.random.randint(0, len(co_metas))
                
                mae_raw_ax, psnr_raw_ax, ssim_raw_ax, ncc_raw_ax = infer_and_save_slice(
                    inr=inr, slice_mod=slice_mod,
                    img_nib=ax_img_nib, img_data=ax_np, metas=ax_metas,
                    slice_axis=ax_slice_axis, slice_index=ax_idx,
                    normalizer=normalizer, stack_id=0,
                    N_ax=N_ax, N_co=N_co,
                    run_dirs=run_dirs, K=K_iter, device=str(device),
                    step=it, normalize_sigma=True, wandb_log=True, rot_mode=rot_mode, style=style
                )
                mae_raw_co, psnr_raw_co, ssim_raw_co, ncc_raw_co = infer_and_save_slice(
                    inr=inr, slice_mod=slice_mod,
                    img_nib=co_img_nib, img_data=co_np, metas=co_metas,
                    slice_axis=co_slice_axis, slice_index=co_idx,
                    normalizer=normalizer, stack_id=1,
                    N_ax=N_ax, N_co=N_co,
                    run_dirs=run_dirs, K=K_iter, device=str(device),
                    step=it, normalize_sigma=True, wandb_log=True, rot_mode=rot_mode, style=style
                )

                mae_avg = (mae_raw_ax+mae_raw_co)*0.5
                psnr_avg = (psnr_raw_ax+psnr_raw_co)*0.5
                ssim_avg = (ssim_raw_ax+ssim_raw_co)*0.5
                ncc_avg = (ncc_raw_ax+ncc_raw_co)*0.5
 
                try:
                    wandb.log({
                        f"Recon/MAE": mae_avg,
                        f"Recon/PSNR": psnr_avg,
                        f"Recon/SSIM": ssim_avg,
                        f"Recon/NCC": ncc_avg,
                    }, step=it)
                except Exception:
                    pass

            if test_batch: break

    # --- 5. Final Evaluation ---
    print("\n=== Final evaluation (full stacks, mean over slices) ===")
    inr.eval(); slice_mod.eval()
    
    K_eval = K
    ax_eval = evaluate_full_stack(
        inr=inr, slice_mod=slice_mod,
        img_nib=ax_img_nib, img_data=ax_np, metas=ax_metas,
        slice_axis=ax_slice_axis, normalizer=normalizer,
        stack_id=0, K=K_eval, device=str(device),
        N_ax=N_ax, N_co=N_co,
        normalize_sigma=True, eval_normalize_gt=True, rot_mode=rot_mode, style=style
    )
    
    co_eval = evaluate_full_stack(
        inr=inr, slice_mod=slice_mod,
        img_nib=co_img_nib, img_data=co_np, metas=co_metas,
        slice_axis=co_slice_axis, normalizer=normalizer,
        stack_id=1, K=K_eval, device=str(device),
        N_ax=N_ax, N_co=N_co,
        normalize_sigma=True, eval_normalize_gt=True, rot_mode=rot_mode, style=style
    )
    
    N_total = ax_eval["n"] + co_eval["n"]
    overall_MAE  = (ax_eval["MAE"]  * ax_eval["n"] + co_eval["MAE"]  * co_eval["n"]) / N_total
    overall_PSNR = (ax_eval["PSNR"] * ax_eval["n"] + co_eval["PSNR"] * co_eval["n"]) / N_total
    overall_SSIM = (ax_eval["SSIM"] * ax_eval["n"] + co_eval["SSIM"] * co_eval["n"]) / N_total
    overall_NCC  = (ax_eval["NCC"]  * ax_eval["n"] + co_eval["NCC"]  * co_eval["n"]) / N_total

    print(f"Axial : MAE={ax_eval['MAE']:.4f}  PSNR={ax_eval['PSNR']:.1f}  SSIM={ax_eval['SSIM']:.2f}  (n={ax_eval['n']})")
    print(f"Coronal: MAE={co_eval['MAE']:.4f}  PSNR={co_eval['PSNR']:.1f}  SSIM={co_eval['SSIM']:.2f}  (n={co_eval['n']})")
    print(f"Overall (weighted): MAE={overall_MAE:.4f}  PSNR={overall_PSNR:.1f}  SSIM={overall_SSIM:.2f}\n")

    if save_to_excel:
        print(f"Appending results to {excel_path}, sheet {excel_sheet}")
        default_name = wandb.run.name if wandb.run is not None else run_dirs.get("stamp", "run")
        row_name = run_name_for_excel or default_name
        row_dict = {
            "name": row_name,
            "axial_MAE":  ax_eval["MAE"], "axial_PSNR": ax_eval["PSNR"], "axial_SSIM": ax_eval["SSIM"], "axial_NCC": ax_eval["NCC"],
            "coronal_MAE":  co_eval["MAE"], "coronal_PSNR": co_eval["PSNR"], "coronal_SSIM": co_eval["SSIM"], "coronal_NCC": co_eval["NCC"],
            "overall_MAE":  overall_MAE, "overall_PSNR": overall_PSNR, "overall_SSIM": overall_SSIM, "overall_NCC": overall_NCC,
        }
        _append_results_to_excel(excel_path=excel_path, sheet_name=excel_sheet, row_dict=row_dict)

    try:
        wandb.run.summary["final/axial_MAE"] = ax_eval["MAE"]
        wandb.run.summary["final/axial_PSNR"] = ax_eval["PSNR"]
        wandb.run.summary["final/axial_SSIM"] = ax_eval["SSIM"]
        wandb.run.summary["final/NCC_axial"] = ax_eval["NCC"]
        wandb.run.summary["final/coronal_MAE"] = co_eval["MAE"]
        wandb.run.summary["final/coronal_PSNR"] = co_eval["PSNR"]
        wandb.run.summary["final/coronal_SSIM"] = co_eval["SSIM"]
        wandb.run.summary["final/NCC_coronal"] = co_eval["NCC"]
        wandb.run.summary["final/overall_MAE"] = overall_MAE
        wandb.run.summary["final/overall_PSNR"] = overall_PSNR
        wandb.run.summary["final/overall_SSIM"] = overall_SSIM
        wandb.run.summary["final/overall_NCC"] = overall_NCC
    except Exception:
        pass

    return inr, slice_mod, normalizer


if __name__ == "__main__":
    # --- Configuration ---
    style = 'hetero'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)
    
    files_path = ""
    files = os.listdir(files_path)
    print(f"Found {len(files)} files in {files_path}")
    
    
    for f in files:
        print(f"RUNNING ON: {f}")
       
        axial_path = f"{files_path}/{f}/Axial.nii.gz"
        coronal_path = f"{files_path}/{f}/Coronal.nii.gz"

        scan_nbr = f

        # Run Config
        cfg = {
            "run_name": f"SRPAN_{scan_nbr}",
            "batch_size": 6000, "lr_inr": 5e-5, "lr_slice": 5e-4, "alpha_iter": 500,
            "K_max": 64, "hidden": 330, "depth": 6, "omega_siren": 30,
            "rot_mode": "rodrigues", "style": style,
            "gating": True, "save_to_excel": True, "excel_path": "results/metrics.xlsx", "excel_sheet": "SRPAN",
        }
     
        wandb.init(
                project="SRPAN",
                name=cfg["run_name"],
                job_type="train_svr",
                config=cfg,
                reinit=True,       
            )
        config = wandb.config


        # Load Data
        ax_vol, ax_aff, _ = load_volume(axial_path)
        co_vol, co_aff, _ = load_volume(coronal_path)
        ax_metas = slice_metadata(ax_vol, ax_aff, 2)
        co_metas = slice_metadata(co_vol, co_aff, 1)
        
        os.makedirs("meta", exist_ok=True)
        save_stack_meta_json(ax_vol, ax_metas, "meta/axial_meta.json", slice_axis=2, path=axial_path)
        save_stack_meta_json(co_vol, co_metas, "meta/coronal_meta.json", slice_axis=1, path=coronal_path)

        run_dirs = make_run_dirs(base_out_dir = "results/SRPAN", run_name=cfg["run_name"])

        inr, slice_mod, _ = train_one_case(
            ax_img_nib=nib.load(axial_path), ax_metas=ax_metas, ax_slice_axis=2,
            co_img_nib=nib.load(coronal_path), co_metas=co_metas, co_slice_axis=1,
            rot_mode=cfg["rot_mode"], excel_sheet=cfg["excel_sheet"],
            alpha_iter=cfg["alpha_iter"], batch_size=cfg["batch_size"],
            device=device, run_dirs=run_dirs, K=cfg["K_max"],
            hidden=cfg["hidden"], depth=cfg["depth"], omega_siren=cfg["omega_siren"],
            save_to_excel=cfg["save_to_excel"], excel_path=cfg["excel_path"],
            run_name_for_excel=cfg["run_name"], style=cfg["style"], gating=cfg["gating"]
        )

        save_path = run_dirs["run_dir"]
        torch.save(inr.state_dict(), os.path.join(save_path, "inr.pth"))
        torch.save(slice_mod.state_dict(), os.path.join(save_path, "slice_module.pth"))
        print(f"Done. Checkpoints saved to {save_path}")

        wandb.finish()