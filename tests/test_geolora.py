import torch
from geolora.geolora import GeoLoRA
from geolora.config import GeoLoRAConfig

B = 2
SEQ = 128


def make_geolora():
    return GeoLoRA(GeoLoRAConfig())


def test_compute_deltas_shape():
    geo = make_geolora()
    depth_map = torch.randn(B, 1, 224, 224)
    h_dict = {}
    cfg = GeoLoRAConfig()
    for layer_idx in range(len(cfg.target_layers)):
        h_dict[layer_idx] = torch.randn(B, SEQ, cfg.d_lm)

    deltas = geo.compute_deltas(depth_map, h_dict)

    assert isinstance(deltas, dict)
    for layer_idx in range(len(cfg.target_layers)):
        assert layer_idx in deltas
        for proj in cfg.target_projections:
            assert proj in deltas[layer_idx]
            d_out = cfg.proj_dims(proj)[1]
            assert deltas[layer_idx][proj].shape == (B, SEQ, d_out)


def test_zero_init_output():
    geo = make_geolora()
    depth_map = torch.randn(B, 1, 224, 224)
    cfg = GeoLoRAConfig()
    h_dict = {i: torch.randn(B, SEQ, cfg.d_lm) for i in range(len(cfg.target_layers))}
    deltas = geo.compute_deltas(depth_map, h_dict)
    for layer_deltas in deltas.values():
        for delta in layer_deltas.values():
            assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)


def test_gradient_flows_to_depth():
    geo = make_geolora()
    depth_map = torch.randn(B, 1, 224, 224, requires_grad=True)
    cfg = GeoLoRAConfig()
    h_dict = {i: torch.randn(B, SEQ, cfg.d_lm) for i in range(len(cfg.target_layers))}
    deltas = geo.compute_deltas(depth_map, h_dict)
    total = sum(d.sum() for ld in deltas.values() for d in ld.values())
    total.backward()
    assert depth_map.grad is not None


def test_param_count():
    geo = make_geolora()
    total = sum(p.numel() for p in geo.parameters())
    assert 8_000_000 < total < 11_000_000, f"Unexpected param count: {total:,}"
