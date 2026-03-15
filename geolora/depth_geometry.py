import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthGeometryNet(nn.Module):
    """
    Processes a raw depth map into a scene geometry descriptor.

    Input:  depth_map R^{B x 1 x H x W}
    Output: z_geo R^{B x d_geo}

    Augments depth with Sobel gradients (surface normal approx)
    before passing through a 3-layer CNN + global pooling + MLP.
    """

    def __init__(self, d_geo: int = 256):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        self.conv1 = nn.Conv2d(3, 64, 7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.norm = nn.LayerNorm(256)
        self.proj = nn.Linear(256, d_geo)

    def _augment_with_gradients(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Concatenate depth with Sobel x/y gradients -> R^{B x 3 x H x W}."""
        grad_x = F.conv2d(depth_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth_map, self.sobel_y, padding=1)
        return torch.cat([depth_map, grad_x, grad_y], dim=1)

    def forward(self, depth_map: torch.Tensor) -> torch.Tensor:
        x = self._augment_with_gradients(depth_map)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = self.pool(x).flatten(1)
        x = F.gelu(self.proj(self.norm(x)))
        return x
