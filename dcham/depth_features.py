import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthFeatureNet(nn.Module):
    """
    Extracts multi-scale spatial features from a raw depth map.

    Input: depth_map R^{B x 1 x H x W} (from P1 or DAv2)
    Output: dict with fine, coarse, global depth features
    """

    def __init__(self, d_depth: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(64, d_depth, 3, stride=2, padding=1)
        self.pool_coarse = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_global = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self, depth_map: torch.Tensor, vit_grid_h: int, vit_grid_w: int
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            depth_map: R^{B x 1 x H x W}
            vit_grid_h: number of ViT patches vertically
            vit_grid_w: number of ViT patches horizontally

        Returns:
            dict with:
              fine:   R^{B x (vit_grid_h * vit_grid_w) x d_depth}
              coarse: R^{B x 16 x d_depth}
              global: R^{B x d_depth}
        """
        x = F.gelu(self.conv1(depth_map))
        x = F.gelu(self.conv2(x))

        fine = F.adaptive_avg_pool2d(x, (vit_grid_h, vit_grid_w))
        fine = fine.flatten(2).transpose(1, 2)

        coarse = self.pool_coarse(x)
        coarse = coarse.flatten(2).transpose(1, 2)

        glob = self.pool_global(x).flatten(1)

        return {"fine": fine, "coarse": coarse, "global": glob}
