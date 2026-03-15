import torch
from dcham.hyper_attention import HyperAttentionHead

B, N, M = 2, 256, 8
D_VIT, D_HEAD, H = 1280, 64, 8
D_DEPTH, NUM_BASIS, RANK = 128, 12, 24


def make_head():
    return HyperAttentionHead(
        d_vit=D_VIT, d_head=D_HEAD, num_heads=H,
        num_basis=NUM_BASIS, rank=RANK, d_depth=D_DEPTH,
    )


def test_output_shape():
    head = make_head()
    visual = torch.randn(B, N, D_VIT)
    depth_global = torch.randn(B, D_DEPTH)
    Q = torch.randn(B, M, H, D_HEAD)
    out = head(visual, depth_global, Q)
    assert out.shape == (B, M, H * D_HEAD)


def test_different_N():
    """Visual token count can vary (dynamic resolution)."""
    head = make_head()
    depth_global = torch.randn(B, D_DEPTH)
    Q = torch.randn(B, M, H, D_HEAD)
    for n in [64, 256, 1024]:
        visual = torch.randn(B, n, D_VIT)
        out = head(visual, depth_global, Q)
        assert out.shape == (B, M, H * D_HEAD)


def test_different_depth_changes_output():
    """Different depth features should produce different outputs."""
    head = make_head()
    visual = torch.randn(B, N, D_VIT)
    Q = torch.randn(B, M, H, D_HEAD)
    d1 = torch.randn(B, D_DEPTH)
    d2 = torch.randn(B, D_DEPTH) * 5
    out1 = head(visual, d1, Q)
    out2 = head(visual, d2, Q)
    assert not torch.allclose(out1, out2, atol=1e-5)


def test_gradient_flows_to_basis_kernels():
    head = make_head()
    visual = torch.randn(B, N, D_VIT)
    depth_global = torch.randn(B, D_DEPTH)
    Q = torch.randn(B, M, H, D_HEAD)
    out = head(visual, depth_global, Q)
    out.sum().backward()
    assert head.V_K.grad is not None
    assert head.U_K.grad is not None
    assert head.V_V.grad is not None


def test_alpha_sums_to_one():
    """Coefficients should be valid convex combination."""
    head = make_head()
    depth_global = torch.randn(1, D_DEPTH)
    ctx = head.context_enc(depth_global)
    alpha_K = torch.softmax(head.coeff_K(ctx).view(1, H, NUM_BASIS), dim=-1)
    sums = alpha_K.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
