# siren.py
import math
import torch
from torch import nn
import torch.nn.functional as F   # <-- add this

class SineLayer(nn.Module):
    """
    A single SIREN layer: y = sin(omega_0 * (Wx + b))

    Paper-faithful initialization:
    - First (is_first=True): W ~ U(-1/in_features, 1/in_features), then use omega_0 in forward.
    - Hidden/other:        W ~ U(-sqrt(6/fan_in)/omega_0, +sqrt(6/fan_in)/omega_0)

    References:
      - Principled init with U(-√6/fan_in, √6/fan_in) and ω0 scaling (supplement §1.3, §1.5). 
      - First-layer frequency factor ω0 to span multiple periods (paper §3.2). 
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = float(omega_0)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer init (keeps inputs roughly uniform -> arcsine after sine)
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)  # paper uses ω0 in forward, not in init
            else:
                # Hidden/other layers: scale by 1/omega_0 to keep activation distribution constant
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.fill_(0.0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_with_intermediate(self, x):
        """Return (activation_after_sin, pre_sin_linear*omega0) for visualization."""
        intermed = self.omega_0 * self.linear(x)
        return torch.sin(intermed), intermed



class Siren(nn.Module):
    """
    Paper-faithful SIREN MLP.

    Args:
        in_features:    input dimension (e.g., 2 for (x,y), 3 for (x,y,t), etc.)
        hidden_features:width of hidden layers
        hidden_layers: number of *hidden* sine layers after the first
        out_features:   output dimension (e.g., 1 for SDF, 3 for RGB)
        outermost_linear: if True, last layer is linear (no sine); else it is a sine layer
        first_omega_0:  ω0 for first layer (default 30)
        hidden_omega_0: ω0 for hidden layers (default 30)

    Returns:
        forward(coords) -> (output, coords.requires_grad_(True))
    """
    def __init__(self, in_dim=3, hidden=330, depth=6, w0_hidden=30.0, outermost_linear=True):

        super().__init__()
        in_features = in_dim
        out_features = 1
        hidden_features = hidden
        hidden_layers = depth
        first_omega_0 = hidden_omega_0 = w0_hidden
        bias = True

        print(f"SIREN: hidden: {hidden} depth: {depth} omega: {w0_hidden}")

        net = []
        # First layer
        net.append(
            SineLayer(
                in_features, hidden_features, bias=bias, is_first=True, omega_0=first_omega_0
            )
        )

        # Hidden sine layers
        for _ in range(hidden_layers):
            net.append(
                SineLayer(
                    hidden_features, hidden_features, bias=bias, is_first=False, omega_0=hidden_omega_0
                )
            )

        # Output layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features, bias=bias)
            with torch.no_grad():
                # Same √6/fan_in, but scaled by 1/hidden_omega_0 to preserve magnitude (supplement §1.5)
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
                if final_linear.bias is not None:
                    final_linear.bias.fill_(0.0)
            net.append(final_linear)
        else:
            net.append(
                SineLayer(
                    hidden_features, out_features, bias=bias, is_first=False, omega_0=hidden_omega_0
                )
            )

        self.net = nn.Sequential(*net)

    def forward(self, coords):
        return self.net(coords)



class SirenHetero(nn.Module):
    """
    SIREN trunk + two heads:
      head_mu:      predicted intensity μ(x)
      head_logvar:  predicted log-variance s(x) = log σ^2(x)
    """
    def __init__(self, in_dim=3, hidden=330, depth=6, w0_hidden=30.0, eps: float = 1e-12):
        super().__init__()

        self.hidden = hidden
        self.depth  = depth
        self.w0     = float(w0_hidden)
        self.eps    = float(eps)  # small positive constant for numerical stability

        print(f"SIREN-HETERO: hidden {hidden} depth {depth} omega {w0_hidden}")

        layers = []
        layers.append(SineLayer(in_dim, hidden, is_first=True, omega_0=self.w0))
        for _ in range(depth):
            layers.append(SineLayer(hidden, hidden, is_first=False, omega_0=self.w0))
        self.trunk = nn.Sequential(*layers)

        def _init_head(h, out_dim=1):
            head = nn.Linear(h, out_dim, bias=True)
            with torch.no_grad():
                bound = math.sqrt(6.0 / h) / self.w0
                head.weight.uniform_(-bound, bound)
                if head.bias is not None:
                    head.bias.fill_(0.0)
            return head

        self.head_mu     = _init_head(hidden, 1)
        self.head_logvar = _init_head(hidden, 1)

    def forward(self, x):
        h   = self.trunk(x)
        mu_raw = self.head_mu(h).squeeze(-1)
        z   = self.head_logvar(h).squeeze(-1)     # (N,)  raw log-variance
        sigma = F.softplus(z) + self.eps                # σ(x) > 0 (smooth positivity)
        s     = 2.0 * torch.log(sigma)                  # s(x) = log σ^2(x)

        return mu_raw, s

  