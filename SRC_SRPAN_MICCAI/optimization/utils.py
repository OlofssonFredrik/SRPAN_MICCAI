# utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from openpyxl import load_workbook
import torch.nn.functional as F

def get_slice_params(
    slice_mod,
    N_ax: int,
    N_co: int,
    device: torch.device,
):
    """
    Returns:
      sig_all  : (N_ax+N_co,) per-slice σ (per chosen normalization)
                  - axial:  all ones (no ω touching axial)
                  - coronal: softmax over coronal only, mean == 1 within coronal
      w_all    : (N_ax+N_co,3)
      t_all    : (N_ax+N_co,3)
    """
    # indices
    sid_ax = torch.zeros(N_ax, dtype=torch.long, device=device) if N_ax > 0 else torch.empty(0, dtype=torch.long, device=device)
    zid_ax = torch.arange(N_ax, dtype=torch.long, device=device) if N_ax > 0 else torch.empty(0, dtype=torch.long, device=device)
    sid_co = torch.ones(N_co,  dtype=torch.long, device=device) if N_co > 0 else torch.empty(0, dtype=torch.long, device=device)
    zid_co = torch.arange(N_co, dtype=torch.long, device=device) if N_co > 0 else torch.empty(0, dtype=torch.long, device=device)

    sid_all = torch.cat([sid_ax, sid_co], dim=0)
    zid_all = torch.cat([zid_ax, zid_co], dim=0)

    out_all = slice_mod(sid_all, zid_all)

    w_all = out_all["w"]
    t_all = out_all["t"]

    # ---- Intensity Scale (C_i) ----
    scale_logits = out_all["scale_logit"].squeeze(-1)
    N_tot = scale_logits.shape[0]
    scale_all = F.softmax(scale_logits, dim=0) * max(N_tot, 1)

    return scale_all, w_all, t_all


class CoordNormalizer:
    def __init__(self, world_min: np.ndarray, world_max: np.ndarray, device="cpu"):
        self.world_min = torch.as_tensor(world_min, dtype=torch.float32, device=device)
        self.world_max = torch.as_tensor(world_max, dtype=torch.float32, device=device)
        self.center = 0.5*(self.world_min + self.world_max)
        self.scale  = 0.5*(self.world_max - self.world_min) + 1e-6
    def world_to_inr(self, xyz):
        return (xyz - self.center) / self.scale
    

def p05(x: torch.Tensor) -> float:
    x = x.detach().reshape(-1)
    return float(torch.quantile(x, 0.05))

def p95(x: torch.Tensor) -> float:
    x = x.detach().reshape(-1)
    return float(torch.quantile(x, 0.95))

def _append_results_to_excel(excel_path: str, sheet_name: str, row_dict: dict):
    """
    Append a single row to an Excel sheet.
    - Creates file/sheet if missing.
    - If sheet exists, reads it, appends the row, and REPLACES the sheet.
    This avoids 'sheet exists' errors and overlay quirks.
    """
    os.makedirs(os.path.dirname(excel_path) or ".", exist_ok=True)

    columns = [
        "name",
        "axial_MAE", "axial_PSNR", "axial_SSIM", "axial_NCC",
        "coronal_MAE", "coronal_PSNR", "coronal_SSIM", "coronal_NCC",
        "overall_MAE", "overall_PSNR", "overall_SSIM", "overall_NCC",
    ]
    row = {c: row_dict.get(c, None) for c in columns}
    df_new = pd.DataFrame([row], columns=columns)

    # Case 1: file doesn't exist -> just write it
    if not os.path.exists(excel_path):
        df_new.to_excel(excel_path, sheet_name=sheet_name, index=False)
        return

    # Case 2: file exists -> try reading the sheet (may not exist yet)
    try:
        df_old = pd.read_excel(excel_path, sheet_name=sheet_name)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        # Replace existing sheet with concatenated data
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            df_all.to_excel(w, sheet_name=sheet_name, index=False)
    except ValueError:
        # Sheet doesn't exist yet -> create it
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a") as w:
            df_new.to_excel(w, sheet_name=sheet_name, index=False)

def tensor_stats(x: torch.Tensor):
    x = x.detach().reshape(-1)
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "med": float(x.median()),
    }



def pose_sigma_stats(sig_all: torch.Tensor,
                     w_all: torch.Tensor,
                     t_all: torch.Tensor,
                     N_ax: int, N_co: int):
    deg_all  = torch.linalg.norm(w_all.detach(), dim=-1) * (180.0/np.pi)
    tmag_all = torch.linalg.norm(t_all.detach(), dim=-1)

    def _robust_stats(x: torch.Tensor):
        x = x.detach().reshape(-1)
        return {
            "mean": float(x.mean()),
            "med":  float(x.median()),
            "p05":  float(torch.quantile(x, 0.05)) if x.numel() else None,
            "p95":  float(torch.quantile(x, 0.95)) if x.numel() else None,
        }

    parts = {}
    # σ stats
    parts["sigma/all"] = _robust_stats(sig_all)
    if N_ax > 0: parts["sigma/ax"] = _robust_stats(sig_all[:N_ax])
    if N_co > 0: parts["sigma/co"] = _robust_stats(sig_all[N_ax:])

    # pose stats (unchanged)
    parts["rot_deg/all"] = tensor_stats(deg_all)
    parts["t_mag/all"]   = tensor_stats(tmag_all)
    if N_ax > 0:
        parts["rot_deg/ax"] = tensor_stats(deg_all[:N_ax])
        parts["t_mag/ax"]   = tensor_stats(tmag_all[:N_ax])
    if N_co > 0:
        parts["rot_deg/co"] = tensor_stats(deg_all[N_ax:])
        parts["t_mag/co"]   = tensor_stats(tmag_all[N_ax:])
    return parts



def _stats_1d(x):
    x = x.detach().reshape(-1)
    return dict(
        mean=float(x.mean()),
        med=float(x.median()),
        p90=float(x.quantile(0.90)),
        mx =float(x.abs().max()),
    )

def log_pose_stats(w_all, t_all, N_ax, N_co, name="", voxel_units="world"):
    """
    w_all: (N_ax+N_co, 3) axis-angle [radians]
    t_all: (N_ax+N_co, 3) translation [same units as renderer/world]
    """
    deg = torch.linalg.norm(w_all.detach(), dim=-1) * (180.0/np.pi)  # (M,)
    tmag = torch.linalg.norm(t_all.detach(), dim=-1)                 # (M,)

    def pr(prefix, v):
        s = _stats_1d(v)
        print(f"{prefix} mean={s['mean']:.3f}, med={s['med']:.3f}, p90={s['p90']:.3f}, max={s['mx']:.3f}")

    if name: print(f"--- Pose stats: {name} ---")
    pr("rot_deg(all)", deg); pr(f"t_{voxel_units}(all)", tmag)

    if N_ax>0:
        pr("rot_deg(ax)", deg[:N_ax]); pr(f"t_{voxel_units}(ax)", tmag[:N_ax])
    if N_co>0:
        pr("rot_deg(co)", deg[N_ax:]); pr(f"t_{voxel_units}(co)", tmag[N_ax:])
