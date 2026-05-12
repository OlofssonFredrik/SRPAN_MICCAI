# renderer.py
import torch
import numpy as np
import math

from src.optimization.utils import CoordNormalizer

def psf_sigmas_paper(zooms_xyz, slice_axis: int):
    """
    Paper's PSF:
      Σ = diag( (1.2*rx/2.355)^2, (1.2*ry/2.355)^2, (rz/2.355)^2 )
    Returns sigma_u, sigma_v, sigma_w aligned with (inplane1, inplane2, through) of THIS stack.
    """
    z = [float(v) for v in zooms_xyz[:3]]
    in_axes = [ax for ax in (0,1,2) if ax != slice_axis]
    rx, ry = z[in_axes[0]], z[in_axes[1]]
    rz     = z[slice_axis]
    s_u = (1.2 * rx) / 2.355
    s_v = (1.2 * ry) / 2.355
    s_w = (rz)       / 2.355
    return s_u, s_v, s_w

def _normalize(v, eps=1e-9):
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / (n + eps)

def build_slice_frame(affine: torch.Tensor, slice_axis: int):
    """
    Build a right-handed, orthonormal slice-local frame (e_u, e_v, e_w) in world space.
    e_w: through-slice unit normal (from the affine's column of 'slice_axis')
    e_u: in-plane unit axis
    e_v: = e_w × e_u  (enforces right-handedness)
    """
    cols = [affine[:3,0], affine[:3,1], affine[:3,2]]  # world-from-voxel columns
    e_w = _normalize(cols[slice_axis])

    in_axes = [ax for ax in (0,1,2) if ax != slice_axis]
    v_u = cols[in_axes[0]]

    # orthonormalize v_u against e_w, then cross for right-handed frame
    e_u = _normalize(v_u - (v_u * e_w).sum(-1, keepdim=True) * e_w)
    e_v = torch.linalg.cross(e_w, e_u)
    e_v = _normalize(e_v)
    return e_u, e_v, e_w

def apply_affine_torch(affine: torch.Tensor, ijk: torch.Tensor):
    """
    affine: (4,4), ijk: (B,3) -> world: (B,3)
    world = ijk @ R^T + t
    """
    R = affine[:3, :3]   # (3,3)
    t = affine[:3,  3]   # (3,)
    return ijk @ R.T + t

def pixel_ij_to_world_batched(affine: torch.Tensor,
                              slice_axis: int,
                              slice_indices: torch.Tensor,   # (B,)
                              ij_pixels: torch.Tensor):       # (B,2)
    """
    Map per-pixel (i,j,s) to world coords. Handles arbitrary slice_axis.
    """
    device = affine.device
    B = ij_pixels.shape[0]
    ijk = torch.zeros((B,3), dtype=torch.float32, device=device)
    in_axes = [ax for ax in (0,1,2) if ax != slice_axis]
    ijk[:, in_axes[0]] = ij_pixels[:, 0]
    ijk[:, in_axes[1]] = ij_pixels[:, 1]
    ijk[:, slice_axis]  = slice_indices.float()
    return apply_affine_torch(affine, ijk)  # (B,3)

def so3_exp_batched(w):
    """
    Stable SO(3) exponential map.
    Uses R = I + a1(θ) [w]_x + a2(θ) [w]_x^2   with
      a1(θ) = sinθ / θ
      a2(θ) = (1 - cosθ) / θ^2
    and Taylor expansions near θ=0 to keep grads non-zero.
    w: (B,3) axis-angle in radians
    returns: (B,3,3) 
    """
    B = w.shape[0]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

    # [w]_x built from raw w (linear in w) → good gradients at 0
    W = torch.zeros((B, 3, 3), dtype=w.dtype, device=w.device)
    W[:, 0, 1] = -wz; W[:, 0, 2] =  wy
    W[:, 1, 0] =  wz; W[:, 1, 2] = -wx
    W[:, 2, 0] = -wy; W[:, 2, 1] =  wx

    theta2 = (w * w).sum(-1, keepdim=True)             # (B,1)
    theta  = torch.sqrt(theta2 + 1e-20)                # (B,1)
    I = torch.eye(3, dtype=w.dtype, device=w.device).unsqueeze(0).expand(B, 3, 3)

    # helper: series for small angles
    small = (theta < 1e-4)
    # a1 = sinθ/θ  ~ 1 - θ^2/6 + θ^4/120
    a1 = torch.sin(theta) / theta
    a1_series = 1.0 - theta2/6.0 + (theta2*theta2)/120.0
    a1 = torch.where(small, a1_series, a1)

    # a2 = (1 - cosθ)/θ^2  ~ 1/2 - θ^2/24 + θ^4/720
    a2 = (1.0 - torch.cos(theta)) / (theta2 + 1e-20)
    a2_series = 0.5 - theta2/24.0 + (theta2*theta2)/720.0
    a2 = torch.where(small, a2_series, a2)

    W2 = W @ W

    R = I + a1.unsqueeze(-1) * W + a2.unsqueeze(-1) * W2

    return R

def se3_apply_batched(points, w, t, pivot):
    """
    Apply per-sample rigid transform about a pivot (slice center).
    points, w, t, pivot: (B,3). Returns (B,3).
    p' = R (p - c) + c + t
    """
    R = so3_exp_batched(w)                     # (B,3,3)
    delta = points - pivot                     # (B,3)
    rot   = torch.einsum('bij,bj->bi', R, delta)  # (B,3)  == R @ delta, per-sample
    return rot + pivot + t

def sample_psf_offsets_world_aniso(
    B: int, K: int,
    e_u: torch.Tensor, e_v: torch.Tensor, e_w: torch.Tensor,
    sigma_u: float, sigma_v: float, sigma_w: float,
    device=None):
    """
    Draw K offsets per pixel from N(0, diag(sigma_u^2, sigma_v^2, sigma_w^2))
    in the slice-local (u,v,w) frame, then map to world.
    Returns (B,K,3).
    """
    device = device or e_u.device
    uvw = torch.zeros((B, K, 3), dtype=torch.float32, device=device)
    if sigma_u > 0: uvw[...,0] = torch.randn((B,K), device=device) * sigma_u
    if sigma_v > 0: uvw[...,1] = torch.randn((B,K), device=device) * sigma_v
    if sigma_w > 0: uvw[...,2] = torch.randn((B,K), device=device) * sigma_w

    eu = e_u.view(1,1,3)
    ev = e_v.view(1,1,3)
    ew = e_w.view(1,1,3)
    offsets_world = uvw[...,0:1]*eu + uvw[...,1:2]*ev + uvw[...,2:3]*ew  # (B,K,3)
    return offsets_world

def se3_apply_batched_from_R(points, R, t, pivot):
    """
    Apply per-sample rigid transform about pivot using a precomputed rotation matrix R.
    points, t, pivot: (B,3), R: (B,3,3) → (B,3)
    p' = R (p - c) + c + t
    """
    delta = points - pivot
    rot = torch.einsum('bij,bj->bi', R, delta)  # (B,3)
    return rot + pivot + t


def simulate_slice_pixels(
    inr: torch.nn.Module,
    affine_np: np.ndarray,
    zooms_xyz,
    slice_axis: int,
    rot_mode,
    slice_indices: torch.Tensor,   # (B,)
    ij_pixels: torch.Tensor,       # (B,2)
    normalizer: CoordNormalizer,
    pose_params: dict,             # {'w':(B,3), 't':(B,3), 'alpha':(B,1)}
    slice_centers_world: torch.Tensor,  # (B,3)
    K: int = 16,
    device: str = "cpu"):

    device = torch.device(device)
    affine = torch.as_tensor(affine_np, dtype=torch.float32, device=device)

    # 1) slice-local frame (right-handed)
    e_u, e_v, e_w = build_slice_frame(affine, slice_axis)

    # 2) pixel centers in world coords (pre-rigid)
    world_pts_pre = pixel_ij_to_world_batched(
        affine, slice_axis, slice_indices.to(device), ij_pixels.to(device)
    )  # (B,3)

    # 3) PSF sigmas (paper)
    s_u, s_v, s_w = psf_sigmas_paper(zooms_xyz, slice_axis)

    # 4) sample PSF offsets in slice-local frame and map to world with E=[e_u,e_v,e_w]
    Bsz = world_pts_pre.shape[0]
    offsets = sample_psf_offsets_world_aniso(
        Bsz, K, e_u, e_v, e_w, s_u, s_v, s_w, device=device
    )  # (B,K,3)

    # 5) pose parameters
    w = pose_params["w"]        # (B,3) or (B,6) if 6D
    t = pose_params["t"]        # (B,3)
    scale = pose_params["scale"]# (B,1)

    # 6) rotation matrix for selected mode
    R = so3_exp_batched(w) 

    # 7) transform pixel centers about slice center
    world_centers_post = se3_apply_batched_from_R(world_pts_pre, R, t, slice_centers_world)


    # 6c) rotate offsets only (no B*K repeat)
    offsets_rot = torch.einsum('bij,bkj->bki', R, offsets)  # (B,K,3)

    # 6d) final sample positions
    samples_world = world_centers_post[:, None, :] + offsets_rot  # (B,K,3)

    # 8) INR query (flatten samples), MC average, scale by alpha (σ)
    samples_norm = normalizer.world_to_inr(samples_world.reshape(-1,3))  # (B*K,3)
    vals = inr(samples_norm).reshape(Bsz, K)                             # (B,K)
    pred = scale.squeeze(-1) * vals.mean(dim=1)                          # (B,)
    return pred


def simulate_slice_pixels_hetero(
    inr: torch.nn.Module,
    affine_np: np.ndarray,
    zooms_xyz,
    slice_axis: int,
    slice_indices: torch.Tensor,       # (B,)
    ij_pixels: torch.Tensor,           # (B,2)
    normalizer: CoordNormalizer,
    pose_params: dict,                 # {'w':(B,3), 't':(B,3), 'alpha':(B,1)}
    slice_centers_world: torch.Tensor, # (B,3)
    K: int = 16,
    device: str = "cpu",
    rot_mode: str = "rodrigues",
):
    device = torch.device(device)
    affine = torch.as_tensor(affine_np, dtype=torch.float32, device=device)

    # 1) local frame
    e_u, e_v, e_w = build_slice_frame(affine, slice_axis)

    # 2) pixel centers (pre-rigid)
    world_pts_pre = pixel_ij_to_world_batched(
        affine, slice_axis, slice_indices.to(device), ij_pixels.to(device)
    )  # (B,3)

    # 3) PSF sigmas
    s_u, s_v, s_w = psf_sigmas_paper(zooms_xyz, slice_axis)

    # 4) PSF samples
    Bsz = world_pts_pre.shape[0]
    offsets = sample_psf_offsets_world_aniso(
        Bsz, K, e_u, e_v, e_w, s_u, s_v, s_w, device=device
    )  # (B,K,3)

    # 5) pose & α
    w     = pose_params["w"]
    t     = pose_params["t"]
    scale = pose_params["scale"]

    # 6) rotation
    R = so3_exp_batched(w)

    # 7) centers after rigid
    world_centers_post = se3_apply_batched_from_R(world_pts_pre, R, t, slice_centers_world)

    # 8) rotate offsets, compose samples
    offsets_rot   = torch.einsum('bij,bkj->bki', R, offsets)  # (B,K,3)
    samples_world = world_centers_post[:, None, :] + offsets_rot

    # 10) INR at K samples
    samples_norm = normalizer.world_to_inr(samples_world.reshape(-1,3))  # (B*K,3)
    mu_k, s_k = inr(samples_norm)     
    mu_k = mu_k.view(Bsz, K)
    s_k  = s_k.view(Bsz, K)  # log σ^2 at sample level

    # 11) PSF aggregationm
    mu_ij_raw = mu_k.mean(dim=1)
    mu_ij = scale.squeeze(-1) * mu_ij_raw

    log_var = (torch.logsumexp(s_k, dim=1) - math.log(K)) + \
                2.0 * torch.log(scale.squeeze(-1) + 1e-12)
    
    return {"mu": mu_ij, "logvar": log_var}                                      
