# wire.py

import torch
from torch import nn
import numpy as np
import math


class WIRELayer(nn.Module):
    """
    Real Gabor activation block (from WIRE, real variant):
      omega = omega0 * (W_f x + b_f)
      sigma = sigma0 * (W_s x + b_s)
      y = cos(omega) * exp(-sigma^2)
    """
    def __init__(self, in_features, out_features,
                 omega0: float = 10.0,
                 sigma0: float = 10.0,
                 bias: bool = True,
                 is_first: bool = False):
        super().__init__()
        self.omega0 = float(omega0)
        self.sigma0 = float(sigma0)
        self.is_first = bool(is_first)

        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        in_dim = self.freqs.weight.shape[1]
        with torch.no_grad():
            if self.is_first:
                # like SIREN first layer
                w = 1.0 / in_dim
                self.freqs.weight.uniform_(-w, w)
                self.scale.weight.uniform_(-w, w)
            else:
                # like SIREN hidden layers, scaled by omega0
                w = math.sqrt(6.0 / in_dim) / max(1.0, self.omega0)
                self.freqs.weight.uniform_(-w, w)
                self.scale.weight.uniform_(-w, w)
            if self.freqs.bias is not None: self.freqs.bias.zero_()
            if self.scale.bias is not None: self.scale.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega = self.omega0 * self.freqs(x)
        sigma = self.sigma0 * self.scale(x)
        return torch.cos(omega) * torch.exp(-(sigma * sigma))


class WIREMLP(nn.Module):
    """
    Real-Gabor WIRE MLP:
      [RealGaborLayer]*depth  ->  Linear(out_features)

    Args mirror your SIREN usage for plug-and-play:
      - hidden: width of hidden layers
      - depth : number of RealGabor layers
      - omega0_first/omega0_hidden: frequency factors
      - sigma0: Gaussian scale (higher = more damping of HF; often helps reduce streaks)
    """
    def __init__(self,
                 in_features: int = 3,
                 out_features: int = 1,
                 hidden: int = 330,
                 depth: int = 6,
                 omega0_first: float = 10.0,
                 omega0_hidden: float = 10.0,
                 sigma0: float = 10.0):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"

        layers = []
        # first layer
        layers.append(WIRELayer(in_features, hidden,
                                     omega0=omega0_first, sigma0=sigma0,
                                     is_first=True))
        # remaining hidden layers
        for _ in range(depth - 1):
            layers.append(WIRELayer(hidden, hidden,
                                         omega0=omega0_hidden, sigma0=sigma0,
                                         is_first=False))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, out_features)

        # SIREN-style head init scaled by omega0_hidden
        with torch.no_grad():
            w = math.sqrt(6.0 / hidden) / max(1.0, omega0_hidden)
            self.head.weight.uniform_(-w, w)
            if self.head.bias is not None:
                self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)


# Example usage
if __name__ == "__main__":
    batch_size = 2
    coords = torch.randn(batch_size, 3)
    model = WIREMLP(in_features=3, hidden_features=256, hidden_layers=3, out_features=1,
                     outermost_linear=True, pos_encode=True)
    output, coords = model(coords)
    print(f"Output shape: {output.shape}")  # [2, 1]
    print(f"Coords shape: {coords.shape}")  # [2, 3]