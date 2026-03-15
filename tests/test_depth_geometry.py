import torch
from geolora.depth_geometry import DepthGeometryNet


def test_output_shape():
    net = DepthGeometryNet(d_geo=256)
    depth_map = torch.randn(2, 1, 448, 672)
    z_geo = net(depth_map)
    assert z_geo.shape == (2, 256)


def test_output_shape_different_resolutions():
    net = DepthGeometryNet(d_geo=256)
    for h, w in [(224, 224), (448, 672), (900, 1600)]:
        depth_map = torch.randn(1, 1, h, w)
        z_geo = net(depth_map)
        assert z_geo.shape == (1, 256), f"Failed for {h}x{w}"


def test_gradient_flows_to_depth_map():
    net = DepthGeometryNet(d_geo=256)
    depth_map = torch.randn(1, 1, 224, 224, requires_grad=True)
    z_geo = net(depth_map)
    z_geo.sum().backward()
    assert depth_map.grad is not None
    assert depth_map.grad.abs().sum() > 0


def test_different_depths_produce_different_outputs():
    net = DepthGeometryNet(d_geo=256)
    d1 = torch.randn(1, 1, 224, 224)
    d2 = torch.randn(1, 1, 224, 224) * 5
    z1 = net(d1)
    z2 = net(d2)
    assert not torch.allclose(z1, z2, atol=1e-5)


def test_sobel_channels():
    """Verify the internal Sobel augmentation produces 3-channel input."""
    net = DepthGeometryNet(d_geo=256)
    depth_map = torch.randn(1, 1, 64, 64)
    augmented = net._augment_with_gradients(depth_map)
    assert augmented.shape == (1, 3, 64, 64)
