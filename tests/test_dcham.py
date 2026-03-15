import torch
from dcham.module import DCHAM
from dcham.config import DCHAMConfig

B = 2


def test_full_forward():
    cfg = DCHAMConfig()
    model = DCHAM(cfg)
    visual_tokens = torch.randn(B, 256, cfg.d_vit)
    depth_map = torch.randn(B, 1, 448, 672)
    text_embeddings = torch.randn(B, 20, cfg.d_lm)
    spatial_tokens = model(visual_tokens, depth_map, text_embeddings, 32, 48)
    assert spatial_tokens.shape == (B, cfg.num_queries, cfg.d_lm)


def test_param_count():
    cfg = DCHAMConfig()
    model = DCHAM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert 4_000_000 < total < 8_000_000, f"Unexpected param count: {total}"


def test_gradient_flow():
    cfg = DCHAMConfig()
    model = DCHAM(cfg)
    visual_tokens = torch.randn(B, 64, cfg.d_vit)
    depth_map = torch.randn(B, 1, 224, 224, requires_grad=True)
    text_embeddings = torch.randn(B, 10, cfg.d_lm)
    out = model(visual_tokens, depth_map, text_embeddings, 16, 16)
    out.sum().backward()
    assert depth_map.grad is not None
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
