import torch
import torch.nn as nn


class DynamicLoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear and adds a per-sample LoRA delta.
    Forward: output = frozen_linear(h) + gate * delta
    Gate initialized to 0 (zero-init) for stable training start.
    """

    def __init__(self, frozen_linear: nn.Linear):
        super().__init__()
        self.frozen_linear = frozen_linear
        for p in self.frozen_linear.parameters():
            p.requires_grad = False
        self.gate = nn.Parameter(torch.zeros(1))
        self._delta = None

    def set_delta(self, delta: torch.Tensor):
        self._delta = delta

    def clear_delta(self):
        self._delta = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        frozen_out = self.frozen_linear(h)
        if self._delta is not None:
            frozen_out = frozen_out + self.gate * self._delta
        return frozen_out
