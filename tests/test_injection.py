import torch
import torch.nn as nn
from geolora.injection import DynamicLoRALinear

B, SEQ = 2, 64
D_IN, D_OUT = 3584, 512


def test_without_delta_matches_original():
    original = nn.Linear(D_IN, D_OUT, bias=True)
    wrapped = DynamicLoRALinear(original)
    h = torch.randn(B, SEQ, D_IN)
    expected = original(h)
    actual = wrapped(h)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_with_delta_adds_gated_offset():
    original = nn.Linear(D_IN, D_OUT, bias=True)
    wrapped = DynamicLoRALinear(original)
    h = torch.randn(B, SEQ, D_IN)
    delta = torch.randn(B, SEQ, D_OUT)
    wrapped.gate.data.fill_(1.0)
    wrapped.set_delta(delta)
    result = wrapped(h)
    expected = original(h) + 1.0 * delta
    assert torch.allclose(result, expected, atol=1e-5)


def test_zero_init_gate_means_no_change():
    original = nn.Linear(D_IN, D_OUT, bias=True)
    wrapped = DynamicLoRALinear(original)
    h = torch.randn(B, SEQ, D_IN)
    delta = torch.randn(B, SEQ, D_OUT) * 100
    wrapped.set_delta(delta)
    result = wrapped(h)
    expected = original(h)
    assert torch.allclose(result, expected, atol=1e-6)


def test_gradient_flows_through_gate():
    original = nn.Linear(D_IN, D_OUT, bias=True)
    wrapped = DynamicLoRALinear(original)
    h = torch.randn(B, SEQ, D_IN)
    delta = torch.randn(B, SEQ, D_OUT, requires_grad=True)
    wrapped.gate.data.fill_(0.5)
    wrapped.set_delta(delta)
    result = wrapped(h)
    result.sum().backward()
    assert wrapped.gate.grad is not None
    assert delta.grad is not None


def test_frozen_linear_no_grad():
    original = nn.Linear(D_IN, D_OUT, bias=True)
    wrapped = DynamicLoRALinear(original)
    h = torch.randn(B, SEQ, D_IN)
    wrapped.gate.data.fill_(1.0)
    wrapped.set_delta(torch.randn(B, SEQ, D_OUT))
    result = wrapped(h)
    result.sum().backward()
    assert original.weight.grad is None
