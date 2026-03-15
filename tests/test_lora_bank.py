import torch
from geolora.lora_bank import LoRABasisBank

B, SEQ = 2, 128
D_IN, D_OUT = 3584, 512
K, R = 6, 16


def make_bank():
    return LoRABasisBank(num_bases=K, lora_rank=R, d_in=D_IN, d_out=D_OUT)


def test_output_shape():
    bank = make_bank()
    h = torch.randn(B, SEQ, D_IN)
    alpha = torch.softmax(torch.randn(B, K), dim=-1)
    delta = bank(h, alpha)
    assert delta.shape == (B, SEQ, D_OUT)


def test_different_alpha_produces_different_output():
    bank = make_bank()
    # Initialise B_bank to non-zero so alpha weighting is observable
    torch.nn.init.xavier_uniform_(bank.B_bank.view(K, -1).reshape(K, D_OUT, R))
    with torch.no_grad():
        for k in range(K):
            torch.nn.init.xavier_uniform_(bank.B_bank[k])
    h = torch.randn(B, SEQ, D_IN)
    a1 = torch.softmax(torch.randn(B, K), dim=-1)
    a2 = torch.softmax(torch.randn(B, K) * 5, dim=-1)
    d1 = bank(h, a1)
    d2 = bank(h, a2)
    assert not torch.allclose(d1, d2, atol=1e-5)


def test_gradient_flows_to_bases():
    bank = make_bank()
    h = torch.randn(B, SEQ, D_IN)
    alpha = torch.softmax(torch.randn(B, K), dim=-1)
    delta = bank(h, alpha)
    delta.sum().backward()
    assert bank.A_bank.grad is not None
    assert bank.B_bank.grad is not None


def test_gradient_flows_through_alpha():
    bank = make_bank()
    h = torch.randn(B, SEQ, D_IN)
    alpha = torch.softmax(torch.randn(B, K, requires_grad=True), dim=-1)
    alpha.retain_grad()  # alpha is a non-leaf; retain_grad() allows .grad to be populated
    delta = bank(h, alpha)
    delta.sum().backward()
    assert alpha.grad is not None


def test_zero_init_B_means_zero_output():
    bank = make_bank()
    h = torch.randn(B, SEQ, D_IN)
    alpha = torch.softmax(torch.randn(B, K), dim=-1)
    delta = bank(h, alpha)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)


def test_param_count():
    bank = make_bank()
    total = sum(p.numel() for p in bank.parameters())
    expected = K * R * D_IN + K * D_OUT * R
    assert total == expected, f"Expected {expected}, got {total}"
