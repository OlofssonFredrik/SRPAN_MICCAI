# slice_module.py
import torch
import torch.nn as nn
import math

from src.models.wire import WIRELayer
from src.models.siren import SineLayer


class _SliceModuleBase(nn.Module):
    """
    Base that handles:
      - per-stack normalized slice positions ∈ [-1, 1]
      - stack embeddings
      - query_all() helper
    Subclasses only need to define the backbone (self.backbone) and heads.
    """
    def __init__(self, n_slices_per_stack, device="cpu", slice_emb_dim=16):
        super().__init__()
        self.n_stacks = len(n_slices_per_stack)
        self.n_slices_per_stack = list(map(int, n_slices_per_stack))
        self.slice_emb_dim = int(slice_emb_dim)

        stack_id_norm_vals=[-1.0, +1.0]

        # Per-stack normalized slice positions in [-1, 1]
        for sid, n in enumerate(self.n_slices_per_stack):
            buf = (torch.linspace(-1.0, 1.0, max(n, 1), device=device)
                   if n > 1 else torch.zeros(1, device=device))
            self.register_buffer(f"idx_norm_{sid}", buf)

        # Normalized stack IDs s ∈ [-1,1] (two stacks → [-1,+1])
        stack_id_norm_vals = torch.linspace(-1.0, 1.0, self.n_stacks, device=device) \
                             if self.n_stacks > 1 else torch.zeros(1, device=device)
        self.register_buffer("stack_id_norm_vals", stack_id_norm_vals)

        # Flatten all slices across stacks: offsets allow mapping (sid, idx) -> flat id
        offsets = []
        running = 0
        for n in self.n_slices_per_stack:
            offsets.append(running)
            running += n
        self.register_buffer("slice_flat_offsets", torch.tensor(offsets, dtype=torch.long, device=device))
        self.total_slices = running if running > 0 else 1
        

    def _gather_slice_norm(self, stack_ids: torch.Tensor, slice_indices: torch.Tensor) -> torch.Tensor:
        """
        Return normalized slice positions z ∈ [-1,1] as shape (B,1)
        for arbitrary mixture of stacks in the batch.
        """
        B = stack_ids.shape[0]
        dev = stack_ids.device
        out = torch.empty((B, 1), dtype=torch.float32, device=dev)
        for sid in range(self.n_stacks):
            m = (stack_ids == sid)
            if not torch.any(m):
                continue
            idxs = slice_indices[m]
            zbuf: torch.Tensor = getattr(self, f"idx_norm_{sid}")  # (n_sid,)
            out[m, 0] = zbuf[idxs]
        return out  # (B,1)

    def _gather_stack_norm(self, stack_ids: torch.Tensor) -> torch.Tensor:
        """
        Return normalized stack id s ∈ [-1,1] as shape (B,1).
        """
        svals = self.stack_id_norm_vals[stack_ids]  # (B,)
        return svals.unsqueeze(-1)                  # (B,1)

    def _gather_slice_flat_ids(self, stack_ids: torch.Tensor, slice_indices: torch.Tensor) -> torch.Tensor:
        """Map (stack_id, slice_idx) to a flat slice id ∈ [0, total_slices)."""
        offsets = self.slice_flat_offsets[stack_ids]  # (B,)
        flat = offsets + slice_indices
        return flat

    def _gather_slice_embedding(self, stack_ids: torch.Tensor, slice_indices: torch.Tensor) -> torch.Tensor:
        """Return per-slice embedding e_i ∈ R^{slice_emb_dim} as (B, slice_emb_dim)."""
        flat_ids = self._gather_slice_flat_ids(stack_ids, slice_indices)
        return self.slice_emb(flat_ids)

class SliceModuleSIREN(_SliceModuleBase):
    def __init__(
        self,
        n_slices_per_stack,
        rot_mode,
        hidden=64,
        w0_first=30.0,
        w0_hidden=30.0,
        device="cpu",
    ):
        super().__init__(n_slices_per_stack, device=device)


        in_dim = 2  # slice & stack indexes
        self.s1 = SineLayer(in_dim, hidden, is_first=True,  omega_0=w0_first)
        self.s2 = SineLayer(hidden, hidden, is_first=False, omega_0=w0_hidden)
        self.rot_mode = rot_mode

        #Heads
        self.motion  = nn.Linear(hidden, 6)  # (w(3), t(3))
        self.scaling = nn.Linear(hidden, 1)  # Intensity Scale (C_i)
  
        # Neutral start
        with torch.no_grad():
            nn.init.zeros_(self.motion.weight);  nn.init.zeros_(self.motion.bias)
            nn.init.zeros_(self.scaling.weight); nn.init.zeros_(self.scaling.bias)

    def forward(self, stack_ids: torch.Tensor, slice_indices: torch.Tensor):
        # Build inputs
        s = self._gather_stack_norm(stack_ids)                    # (B,1)
        z = self._gather_slice_norm(stack_ids, slice_indices)     # (B,1)
 
        x = torch.cat([s, z], dim=-1)                          # (B, 2)

        # Backbone
        h = self.s1(x)
        h = self.s2(h)

        # Heads
        m = self.motion(h)   # (B,6) 
        s = self.scaling(h)


        w = m[..., 0:3]
        t = m[..., 3:6]

        return {
            "w": w,
            "t": t,
            "scale_logit": s[..., 0:1],
        }
            
                    



class SliceModuleWIRE(_SliceModuleBase):
    """
    Same heads/interface as SIREN version, but backbone uses 2 WIRE layers.
    """
    def __init__(
        self,
        n_slices_per_stack,
        rot_mode,
        TUM_style=False,
        hidden=256,
        emb_dim=4,
        w0_first=10.0,
        w0_hidden=10.0,
        sigma=10.0,
        device="cpu",
    ):
        super().__init__(n_slices_per_stack, device=device)


        self.TUM_style = TUM_style
        if self.TUM_style:
            hidden = 256
        print(f"USING WIRE")
        print(f"TUM_style={self.TUM_style}")


        #in_dim = 2 + self.slice_emb_dim  # s, z, e_slice(16) => 18
        in_dim = 2  # s, z  NO SLICE EMBEDDING

        # Backbone: 2 WIRE layers
        self.w1 = WIRELayer(in_dim, hidden, is_first=True)
        self.w2 = WIRELayer(hidden, hidden)

        self.rot_mode = rot_mode

        #Heads
        self.motion  = nn.Linear(hidden, 6)  # (w(3), t(3))
        if self.TUM_style:
            self.outlier = nn.Linear(hidden, 2)  # sigma and omega
        else:
            self.outlier = nn.Linear(hidden, 1)  # sigma only

        # Neutral start
        with torch.no_grad():
            nn.init.zeros_(self.motion.weight);  nn.init.zeros_(self.motion.bias)
            nn.init.zeros_(self.outlier.weight); nn.init.zeros_(self.outlier.bias)

    def forward(self, stack_ids: torch.Tensor, slice_indices: torch.Tensor):
        # Build inputs
        s = self._gather_stack_norm(stack_ids)                    # (B,1)
        z = self._gather_slice_norm(stack_ids, slice_indices)     # (B,1)
        #e = self._gather_slice_embedding(stack_ids, slice_indices)  # (B,16)
        #x = torch.cat([s, z, e], dim=-1)                          # (B, 18)
        x = torch.cat([s, z], dim=-1)                          # (B, 2)  NO SLICE EMBEDDING

        # Backbone
        h = self.w1(x)
        h = self.w2(h)

        # Heads
        m = self.motion(h)   # (B,6) 
        o = self.outlier(h)  # (B,1) NOW ONLY SIGMA


        w = m[..., 0:3]
        t = m[..., 3:6]

        if not self.TUM_style:
            return {
                "w": w,
                "t": t,
                "sigma_logit": o,
            }
        else:
            return {
            "w": w,
            "t": t,
            "sigma_logit": o[..., 0:1],
            "omega_logit": o[..., 1:2],
        }