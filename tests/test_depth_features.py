import torch
from dcham.depth_features import DepthFeatureNet


def test_output_shapes():
    net = DepthFeatureNet(d_depth=128)
    depth_map = torch.randn(1, 1, 448, 672)
    vit_grid_h, vit_grid_w = 32, 48
    feats = net(depth_map, vit_grid_h, vit_grid_w)
    assert feats["fine"].shape == (1, vit_grid_h * vit_grid_w, 128)
    assert feats["coarse"].shape == (1, 16, 128)
    assert feats["global"].shape == (1, 128)


def test_different_resolutions():
    net = DepthFeatureNet(d_depth=128)
    for h, w in [(224, 224), (448, 672), (896, 1344)]:
        depth_map = torch.randn(1, 1, h, w)
        grid_h, grid_w = h // 14, w // 14
        feats = net(depth_map, grid_h, grid_w)
        assert feats["fine"].shape[1] == grid_h * grid_w


def test_gradient_flow():
    net = DepthFeatureNet(d_depth=128)
    depth_map = torch.randn(1, 1, 224, 224, requires_grad=True)
    feats = net(depth_map, 16, 16)
    loss = feats["fine"].sum() + feats["coarse"].sum() + feats["global"].sum()
    loss.backward()
    assert depth_map.grad is not None
