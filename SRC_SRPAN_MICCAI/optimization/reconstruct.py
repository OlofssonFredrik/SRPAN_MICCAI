# reconstruct.py

from typing import Tuple

from src.optimization.train import joint_robust_normalize
import numpy as np
import nibabel as nib
import torch

from src.optimization.utils import CoordNormalizer
from src.data_processing.dataset import slice_metadata
from src.models.siren import SirenHetero, Siren
from src.models.slice_module import SliceModuleSIREN

import numpy as np
import torch
import nibabel as nib

from typing import Tuple

from src.optimization.inference import evaluate_full_stack



def _load_state_dict_strip_module(sd):
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def make_normalizer_from_two_stacks(ax_nii, co_nii, device: str = "cpu"):
    ax_vol = ax_nii.get_fdata().astype(np.float32)
    co_vol = co_nii.get_fdata().astype(np.float32)
    ax_meta = slice_metadata(ax_vol, ax_nii.affine, slice_axis=2)  # axial Z
    co_meta = slice_metadata(co_vol, co_nii.affine, slice_axis=1)  # coronal Y

    pts = []
    for m in ax_meta + co_meta:
        pts.extend(m.fov_corners_world)
    P = np.asarray(pts, dtype=np.float32)
    world_min, world_max = P.min(0), P.max(0)
    print(
        "CoordNormalizer (inference, UNION bbox): world_min =",
        world_min,
        "world_max =",
        world_max,
    )
    return CoordNormalizer(world_min, world_max, device=device)


def world_bbox_from_stacks_intersection(ax_nii, co_nii):
    ax_vol = ax_nii.get_fdata().astype(np.float32)
    co_vol = co_nii.get_fdata().astype(np.float32)

    ax_meta = slice_metadata(ax_vol, ax_nii.affine, slice_axis=2)
    co_meta = slice_metadata(co_vol, co_nii.affine, slice_axis=1)

    pts_ax = np.asarray(
        [p for m in ax_meta for p in m.fov_corners_world], dtype=np.float32
    )
    pts_co = np.asarray(
        [p for m in co_meta for p in m.fov_corners_world], dtype=np.float32
    )

    wmin_ax, wmax_ax = pts_ax.min(0), pts_ax.max(0)
    wmin_co, wmax_co = pts_co.min(0), pts_co.max(0)

    wmin_int = np.maximum(wmin_ax, wmin_co)
    wmax_int = np.minimum(wmax_ax, wmax_co)
    print("Intersection bbox: world_min =", wmin_int, "world_max =", wmax_int)
    return wmin_int.astype(np.float32), wmax_int.astype(np.float32)


def crop_volume_to_world_bbox(
    vol: np.ndarray, aff: np.ndarray, wmin_crop: np.ndarray, wmax_crop: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = float(aff[0, 0]), float(aff[1, 1]), float(aff[2, 2])
    origin = aff[:3, 3].astype(np.float32)

    eps = 1e-3
    i_min = max(0, int(np.floor((wmin_crop[0] - origin[0]) / sx - eps)))
    i_max = min(vol.shape[0] - 1, int(np.ceil((wmax_crop[0] - origin[0]) / sx + eps)))
    j_min = max(0, int(np.floor((wmin_crop[1] - origin[1]) / sy - eps)))
    j_max = min(vol.shape[1] - 1, int(np.ceil((wmax_crop[1] - origin[1]) / sy + eps)))
    k_min = max(0, int(np.floor((wmin_crop[2] - origin[2]) / sz - eps)))
    k_max = min(vol.shape[2] - 1, int(np.ceil((wmax_crop[2] - origin[2]) / sz + eps)))

    vol_c = vol[i_min : i_max + 1, j_min : j_max + 1, k_min : k_max + 1]

    aff_c = aff.copy()
    aff_c[:3, 3] = origin + np.array(
        [i_min * sx, j_min * sy, k_min * sz], dtype=np.float32
    )

    print("Cropped volume shape:", vol_c.shape)
    return vol_c, aff_c



@torch.no_grad()
def reconstruct_isotropic_volume(
    *,
    inr,
    normalizer,
    spacing_xyz=(1.0, 1.0, 1.0),
    device="cuda",
    chunk_voxels=1_500_000,
    stacks_for_mask=None,
    crop_outside_fov=True,
    inner_margin_mm: float = 0.0,
):
    device = torch.device(device)

    # ----- world grid -----
    wmin = normalizer.world_min.detach().cpu().numpy().astype(np.float32)
    wmax = normalizer.world_max.detach().cpu().numpy().astype(np.float32)
    sx, sy, sz = map(float, spacing_xyz)
    xs = np.arange(wmin[0], wmax[0] + 0.5 * sx, sx, dtype=np.float32)
    ys = np.arange(wmin[1], wmax[1] + 0.5 * sy, sy, dtype=np.float32)
    zs = np.arange(wmin[2], wmax[2] + 0.5 * sz, sz, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Nx, Ny, Nz = X.shape
    N = Nx * Ny * Nz

    pts_world = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
    pts_world_t = torch.from_numpy(pts_world).to(device)

    # robust world -> INR mapping
    if hasattr(normalizer, "world_to_inr"):
        pts_inr = normalizer.world_to_inr(pts_world_t)
    else:
        wmin_t = normalizer.world_min.to(pts_world_t.device)
        wmax_t = normalizer.world_max.to(pts_world_t.device)
        pts_inr = 2.0 * (pts_world_t - wmin_t) / (wmax_t - wmin_t + 1e-8) - 1.0

    # ----- evaluate INR -----
    vals = np.zeros((N,), dtype=np.float32)
    inr.eval()
    start = 0
    while start < N:
        end = min(start + chunk_voxels, N)
        pred = inr(pts_inr[start:end])
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        vals[start:end] = pred.squeeze(-1).detach().float().cpu().numpy()
        start = end
    vol = vals.reshape(Nx, Ny, Nz).astype(np.float32)

    # ----- affine for this grid -----
    aff = np.eye(4, dtype=np.float32)
    aff[0, 0], aff[1, 1], aff[2, 2] = sx, sy, sz
    aff[:3, 3] = wmin

    # ----- optional rectangular crop to intersection of both stacks' FOV -----
    if stacks_for_mask is not None and crop_outside_fov:
        ax_nii, co_nii = stacks_for_mask
        wmin_crop, wmax_crop = world_bbox_from_stacks_intersection(ax_nii, co_nii)

        # OPTIONAL: tighten crop with an inner margin in mm
        if inner_margin_mm > 0:
            wmin_crop = wmin_crop + inner_margin_mm
            wmax_crop = wmax_crop - inner_margin_mm

        vol, aff = crop_volume_to_world_bbox(vol, aff, wmin_crop, wmax_crop)

    vol[vol < 0.0] = 0.0

    nii = nib.Nifti1Image(vol, aff)
    return vol, aff, nii

@torch.no_grad()
def reconstruct_on_coronal_grid_inr_only(
    *,
    inr,
    normalizer,
    coronal_nii: nib.Nifti1Image,
    device: str = "cuda",
    chunk_voxels: int = 1_500_000,
    mask_from_coronal: bool = True,
    background_value: float = 0.0,
    inner_margin_mm: float = 0.0,   # <--- NEW
):
    device = torch.device(device)

    co_data = coronal_nii.get_fdata().astype(np.float32)
    aff = coronal_nii.affine.astype(np.float32)
    Sx, Sy, Sz = co_data.shape

    # -------- build ijk grid for the FULL coronal volume --------
    i = np.arange(Sx, dtype=np.float32)
    j = np.arange(Sy, dtype=np.float32)
    k = np.arange(Sz, dtype=np.float32)
    I, J, K = np.meshgrid(i, j, k, indexing="ij")   # (Sx, Sy, Sz)
    ijk = np.stack([I, J, K], axis=-1).reshape(-1, 3)   # (N,3)
    N = ijk.shape[0]

    # -------- build mask --------
    if mask_from_coronal:
        mask3d = (co_data > 0.0)
        mask = mask3d.reshape(-1)
        idx_eval = np.where(mask)[0]
    else:
        mask = None
        idx_eval = np.arange(N, dtype=np.int64)

    print(f"[Coronal] Total voxels: {N}, evaluating INR at {len(idx_eval)} positions")

    # -------- ijk -> world coords using the coronal affine --------
    R = aff[:3, :3]  # (3,3)
    t = aff[:3, 3]   # (3,)
    Pw_all = ijk @ R.T + t         # (N,3) world coords of all voxels
    Pw = Pw_all[idx_eval].astype(np.float32)  # (Neval,3)

    # -------- world -> INR coords (same normalizer as training) --------
    Pw_t = torch.from_numpy(Pw).to(device)
    if hasattr(normalizer, "world_to_inr"):
        Pinr = normalizer.world_to_inr(Pw_t)
    else:
        wmin_t = normalizer.world_min.to(device)
        wmax_t = normalizer.world_max.to(device)
        Pinr = 2.0 * (Pw_t - wmin_t) / (wmax_t - wmin_t + 1e-8) - 1.0

    # -------- evaluate INR in chunks --------
    vals_eval = np.zeros((len(idx_eval),), dtype=np.float32)
    inr.eval()
    start = 0
    while start < len(idx_eval):
        end = min(start + chunk_voxels, len(idx_eval))
        out = inr(Pinr[start:end])
        if isinstance(out, (tuple, list)):  # SirenHetero: (mu, logvar)
            out = out[0]
        vals_eval[start:end] = out.squeeze(-1).detach().float().cpu().numpy()
        start = end

    # -------- scatter into full volume --------
    vol_flat = np.full((N,), background_value, dtype=np.float32)
    vol_flat[idx_eval] = vals_eval
    vol = vol_flat.reshape(Sx, Sy, Sz)

    # -------- OPTIONAL: crop away inner_margin_mm in world space --------
    aff_out = coronal_nii.affine
    if inner_margin_mm > 0.0:
        wmin_full = Pw_all.min(axis=0).astype(np.float32)
        wmax_full = Pw_all.max(axis=0).astype(np.float32)

        wmin_crop = wmin_full + inner_margin_mm
        wmax_crop = wmax_full - inner_margin_mm

        vol, aff_out = crop_volume_to_world_bbox(vol, aff_out, wmin_crop, wmax_crop)

    nii = nib.Nifti1Image(vol.astype(np.float32), aff_out, header=coronal_nii.header)

    print(
        f"[Coronal recon] stats: "
        f"min={vol.min():.3f} max={vol.max():.3f} "
        f"mean={vol.mean():.3f} std={vol.std():.3f}"
    )
    return vol, aff_out, nii

@torch.no_grad()
def reconstruct_on_axial_grid_inr_only(
    *,
    inr,
    normalizer,
    axial_nii: nib.Nifti1Image,
    device: str = "cuda",
    chunk_voxels: int = 1_500_000,
    mask_from_axial: bool = True,
    background_value: float = 0.0,
    inner_margin_mm: float = 0.0,   # <--- NEW
):
    """
    Evaluate ONLY the INR on the exact voxel centers of the axial volume.

    - Output shape == (possibly cropped) axial grid
    - Output affine updated if inner_margin_mm > 0
    """
    device = torch.device(device)

    ax_data = axial_nii.get_fdata().astype(np.float32)
    aff = axial_nii.affine.astype(np.float32)
    Sx, Sy, Sz = ax_data.shape

    # -------- build ijk grid for the axial volume --------
    i = np.arange(Sx, dtype=np.float32)
    j = np.arange(Sy, dtype=np.float32)
    k = np.arange(Sz, dtype=np.float32)
    I, J, K = np.meshgrid(i, j, k, indexing="ij")  # (Sx,Sy,Sz) each
    ijk = np.stack([I, J, K], axis=-1).reshape(-1, 3)          # (N,3)
    N = ijk.shape[0]

    # optional mask: only query inside the acquired FOV
    if mask_from_axial:
        mask = (ax_data > 0.0).reshape(-1)  # (N,)
        idx_eval = np.where(mask)[0]
    else:
        mask = None
        idx_eval = np.arange(N, dtype=np.int64)

    print(f"Total voxels: {N}, evaluating INR at {len(idx_eval)} positions")

    # -------- ijk -> world coords using the axial affine --------
    R = aff[:3, :3]        # (3,3)
    t = aff[:3, 3]         # (3,)
    Pw_all = ijk @ R.T + t  # (N,3) world coords of all voxels
    Pw = Pw_all[idx_eval].astype(np.float32)  # (Neval,3)

    # -------- world -> INR coords (same normalizer as training) --------
    Pw_t = torch.from_numpy(Pw).to(device)
    if hasattr(normalizer, "world_to_inr"):
        Pinr = normalizer.world_to_inr(Pw_t)
    else:
        # fallback: linear map from [world_min, world_max] -> [-1,1]
        wmin_t = normalizer.world_min.to(device)
        wmax_t = normalizer.world_max.to(device)
        Pinr = 2.0 * (Pw_t - wmin_t) / (wmax_t - wmin_t + 1e-8) - 1.0

    # -------- evaluate INR in chunks --------
    vals_eval = np.zeros((len(idx_eval),), dtype=np.float32)
    inr.eval()
    start = 0
    while start < len(idx_eval):
        end = min(start + chunk_voxels, len(idx_eval))
        out = inr(Pinr[start:end])
        # SirenHetero returns (mu, logvar) – take mu
        if isinstance(out, (tuple, list)):
            out = out[0]
        vals_eval[start:end] = (
            out.squeeze(-1).detach().float().cpu().numpy().astype(np.float32)
        )
        start = end

    # -------- scatter into full volume --------
    vol_flat = np.full((N,), background_value, dtype=np.float32)
    vol_flat[idx_eval] = vals_eval
    vol = vol_flat.reshape(Sx, Sy, Sz)

    # -------- OPTIONAL: crop away inner_margin_mm in world space --------
    aff_out = axial_nii.affine
    if inner_margin_mm > 0.0:
        wmin_full = Pw_all.min(axis=0).astype(np.float32)
        wmax_full = Pw_all.max(axis=0).astype(np.float32)

        wmin_crop = wmin_full + inner_margin_mm
        wmax_crop = wmax_full - inner_margin_mm

        vol, aff_out = crop_volume_to_world_bbox(vol, aff_out, wmin_crop, wmax_crop)

    nii = nib.Nifti1Image(vol.astype(np.float32), aff_out, header=axial_nii.header)

    print(
        f"Recon (INR-only axial grid) stats: "
        f"min={vol.min():.3f} max={vol.max():.3f} "
        f"mean={vol.mean():.3f} std={vol.std():.3f}"
    )
    return vol, aff_out, nii


#Reconstruct original data
"""
    out_nii_path = f"results/reconstruction_study/{run_name}_axial.nii.gz"
    

    vol_ax, aff_ax, nii_ax = reconstruct_on_axial_grid_inr_only(
    inr=inr,
    normalizer=normalizer,
    axial_nii=ax_nii,
    device=device,
    chunk_voxels=1_500_000,
    mask_from_axial=True,
    background_value=0.0,
)

    nib.save(nii, out_nii_path)
    print("Saved reconstructed volume to:", out_nii_path)


    out_nii_path_co = f"results/reconstruction_study/{run_name}_coronal.nii.gz"
    

    vol_co, aff_co, nii_co = reconstruct_on_coronal_grid_inr_only(
    inr=inr,
    normalizer=normalizer,
    coronal_nii=co_nii,
    device=device,
    chunk_voxels=1_500_000,
    mask_from_coronal=True,
    background_value=0.0,
)

    nib.save(nii_co, out_nii_path_co)
    print("Saved reconstructed coronal volume to:", out_nii_path_co)
    """

# --------------------------------------------------------------------------
# main reconstruction
# --------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"


    run_name = ""

    results_folder = "results/SRPAN"
    
    axial_scan_path = f"{results_folder}/{run_name}/axial_preprocessed.nii.gz"
    coronal_scan_path = f"{results_folder}/{run_name}/coronal_preprocessed.nii.gz"
    
    ax_nii = nib.load(axial_scan_path)
    co_nii = nib.load(coronal_scan_path)

    ax_np = ax_nii.get_fdata().astype(np.float32)
    ax_slice_axis = 2
    ax_metas = slice_metadata(ax_np, ax_nii.affine, slice_axis=ax_slice_axis)
    N_ax = len(ax_metas)
    co_np = co_nii.get_fdata().astype(np.float32)
    co_slice_axis = 1
    co_metas = slice_metadata(co_np, co_nii.affine, slice_axis=co_slice_axis)
    N_co = len(co_metas)
    rot_mode = "rodrigues"

    #margin = 5  # in mm

    inr_ckpt_path = f"{results_folder}/{run_name}/inr.pth"
    out_nii_path = f"{results_folder}/{run_name}/3D.nii.gz"

    print("Loading pretrained INR:", inr_ckpt_path)
    inr = SirenHetero().to(device)
    slice_mod = SliceModuleSIREN(
        n_slices_per_stack=[len(ax_metas), len(co_metas)],
        hidden=64, device=device, rot_mode=rot_mode
    ).to(device)
    #inr = Siren().to(device)
    sd = torch.load(inr_ckpt_path, map_location=device)
    sd = _load_state_dict_strip_module(sd)
    inr.load_state_dict(sd, strict=True)
    inr.eval()

    #load slicemodule
    sd_slice = torch.load(inr_ckpt_path.replace("inr.pth", "slice_module.pth"), map_location=device)
    sd_slice = _load_state_dict_strip_module(sd_slice)
    slice_mod.load_state_dict(sd_slice, strict=True)
    slice_mod.eval()

    
    #ax_np, co_np = joint_robust_normalize(ax_nii.get_fdata(), co_nii.get_fdata(), p_min=0.5, p_max=99.5)
    #if load from preprocessed files, no need normalisation
    ax_np = ax_nii.get_fdata().astype(np.float32)
    co_np = co_nii.get_fdata().astype(np.float32)



    


    normalizer = make_normalizer_from_two_stacks(ax_nii, co_nii, device=device)


    vol, aff, _ = reconstruct_isotropic_volume(
    inr=inr,
    normalizer=normalizer,
    spacing_xyz=(1, 1, 1),
    device=device,
    stacks_for_mask=(ax_nii, co_nii),
    crop_outside_fov=True,
)

    nii = nib.Nifti1Image(vol.astype(np.float32), aff)
    nib.save(nii, out_nii_path)
    print("Saved reconstructed volume to:", out_nii_path)

    
   
    ax_eval = evaluate_full_stack(
        inr=inr,
        slice_mod=slice_mod,
        img_nib=ax_nii,
        img_data=ax_np,
        metas=ax_metas,
        slice_axis=ax_slice_axis,
        normalizer=normalizer,
        stack_id=0,
        N_ax=N_ax,
        N_co=N_co,
        rot_mode=rot_mode,
        style="hetero",
        K=64,
        device=device,
    )

    co_eval = evaluate_full_stack(
        inr=inr,
        slice_mod=slice_mod,
        img_nib=co_nii,
        img_data=co_np,
        metas=co_metas,
        slice_axis=co_slice_axis,
        normalizer=normalizer,
        stack_id=1,
        N_ax=N_ax,
        N_co=N_co,
        rot_mode=rot_mode,
        style="hetero",
        K=64,
        device=device,
    )

    N_total = ax_eval["n"] + co_eval["n"]
    overall_MAE  = (ax_eval["MAE"]  * ax_eval["n"] + co_eval["MAE"]  * co_eval["n"]) / N_total
    overall_PSNR = (ax_eval["PSNR"] * ax_eval["n"] + co_eval["PSNR"] * co_eval["n"]) / N_total
    overall_SSIM = (ax_eval["SSIM"] * ax_eval["n"] + co_eval["SSIM"] * co_eval["n"]) / N_total
    overall_NCC  = (ax_eval["NCC"]  * ax_eval["n"] + co_eval["NCC"]  * co_eval["n"]) / N_total

    print(f"Axial : MAE={ax_eval['MAE']:.4f}  PSNR={ax_eval['PSNR']:.1f}  SSIM={ax_eval['SSIM']:.2f}  (n={ax_eval['n']})")
    print(f"Coronal: MAE={co_eval['MAE']:.4f}  PSNR={co_eval['PSNR']:.1f}  SSIM={co_eval['SSIM']:.2f}  (n={co_eval['n']})")
    print(f"Overall (weighted): MAE={overall_MAE:.4f}  PSNR={overall_PSNR:.1f}  SSIM={overall_SSIM:.2f}\n")



