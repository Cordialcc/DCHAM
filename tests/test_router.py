import torch
from geolora.router import GeometryRouter

B = 2
D_GEO = 256
K = 6
L = 8


def make_router():
    return GeometryRouter(d_geo=D_GEO, num_bases=K, num_layers=L)


def test_output_shape():
    router = make_router()
    z_geo = torch.randn(B, D_GEO)
    alphas = router(z_geo)
    assert isinstance(alphas, dict)
    assert len(alphas) == L
    for layer_idx, alpha in alphas.items():
        assert alpha.shape == (B, K), f"Layer {layer_idx}: {alpha.shape}"


def test_softmax_sums_to_one():
    router = make_router()
    z_geo = torch.randn(B, D_GEO)
    alphas = router(z_geo)
    for alpha in alphas.values():
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_different_z_geo_produces_different_alpha():
    router = make_router()
    z1 = torch.randn(B, D_GEO)
    z2 = torch.randn(B, D_GEO) * 5
    a1 = router(z1)
    a2 = router(z2)
    for k in a1:
        assert not torch.allclose(a1[k], a2[k], atol=1e-5)


def test_different_layers_can_get_different_alpha():
    router = make_router()
    z_geo = torch.randn(1, D_GEO)
    alphas = router(z_geo)
    layer_keys = list(alphas.keys())
    differ = any(
        not torch.allclose(alphas[layer_keys[0]], alphas[layer_keys[i]], atol=1e-5)
        for i in range(1, len(layer_keys))
    )
    assert differ, "All layers produced identical alpha"


def test_gradient_flows():
    router = make_router()
    z_geo = torch.randn(B, D_GEO, requires_grad=True)
    alphas = router(z_geo)
    total = sum(a.sum() for a in alphas.values())
    total.backward()
    assert z_geo.grad is not None
