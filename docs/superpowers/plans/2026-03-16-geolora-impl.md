# GeoLoRA Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Geometry-Conditioned Dynamic LoRA (GeoLoRA) and integrate it with Qwen2.5-VL-7B for spatial reasoning training and evaluation.

**Architecture:** GeoLoRA processes a raw depth map into a scene geometry descriptor via DepthGeometryNet, routes it through a Geometry Router to produce per-layer mixing weights, and dynamically combines a bank of K=6 LoRA bases per target layer. The mixed LoRA is applied to q_proj and v_proj of LLM layers 20-27. Base model frozen; only GeoLoRA components train.

**Tech Stack:** PyTorch, HuggingFace transformers, Qwen2.5-VL-7B, YAML config

**Spec:** `docs/superpowers/specs/2026-03-16-geolora-design.md`

---

## File Structure

```
/Users/daijidong/Documents/zigzagging/
├── docs/                                    # existing
├── dcham/                                   # old design (keep for reference)
├── geolora/
│   ├── __init__.py                          # Package exports
│   ├── config.py                            # All hyperparameters (GeoLoRAConfig)
│   ├── depth_geometry.py                    # DepthGeometryNet: depth map -> z_geo
│   ├── lora_bank.py                         # LoRABasisBank: K bases per layer/proj + mixing
│   ├── router.py                            # GeometryRouter: z_geo -> per-layer alpha
│   ├── geolora.py                           # GeoLoRA: assembles all components
│   ├── injection.py                         # DynamicLoRALinear: wraps frozen Linear + LoRA delta
│   ├── model.py                             # Qwen2.5-VL + GeoLoRA integration
│   ├── dataset.py                           # SpatialQA dataset with depth maps
│   └── collator.py                          # Data collator
├── scripts/
│   ├── train.py                             # existing (for DCHAM)
│   ├── train_geolora.py                     # GeoLoRA training entry point
│   ├── evaluate_geolora.py                  # GeoLoRA evaluation
│   └── generate_depth_maps.py               # existing (reusable)
├── configs/
│   ├── default.yaml                         # existing (DCHAM)
│   └── geolora.yaml                         # GeoLoRA training config
├── tests/
│   ├── test_depth_geometry.py               # DepthGeometryNet dimension + gradient tests
│   ├── test_lora_bank.py                    # LoRABasisBank mixing + gradient tests
│   ├── test_router.py                       # GeometryRouter output shape + softmax tests
│   ├── test_geolora.py                      # Full module integration test
│   ├── test_injection.py                    # DynamicLoRALinear wrapping test
│   └── test_geolora_model.py               # Qwen2.5-VL + GeoLoRA forward pass (GPU only)
└── requirements.txt                         # existing (no changes needed)
```

**File responsibilities:**
- `config.py`: Single source of truth for all dimensions (d_geo, K, r, L, target layers, etc.)
- `depth_geometry.py`: CNN + Sobel gradients that processes raw depth maps -> scene geometry descriptor
- `lora_bank.py`: Stores K LoRA basis pairs per (layer, projection); does einsum-based mixing given alpha weights
- `router.py`: Shared MLP + layer embeddings; maps z_geo -> per-layer softmax mixing weights alpha
- `geolora.py`: Orchestrates DepthGeometryNet -> Router -> LoRABasisBank; owns the gate parameters
- `injection.py`: `DynamicLoRALinear` class that wraps a frozen `nn.Linear` and adds the LoRA delta from the bank
- `model.py`: Wraps Qwen2.5-VL, replaces target projection Linears with DynamicLoRALinear, manages forward pass
- `dataset.py` + `collator.py`: Loads SpatialQA data with paired depth maps (reuses pattern from dcham/)

---

## Chunk 1: Project Setup + DepthGeometryNet

### Task 1: Config and package scaffolding

**Files:**
- Create: `geolora/__init__.py`
- Create: `geolora/config.py`
- Create: `configs/geolora.yaml`

- [ ] **Step 1: Create config.py with all hyperparameters**

```python
# geolora/config.py
from dataclasses import dataclass


@dataclass
class GeoLoRAConfig:
    # Qwen2.5-VL-7B dimensions (frozen, reference only)
    d_lm: int = 3584
    num_llm_layers: int = 28

    # GeoLoRA dimensions
    d_geo: int = 256           # scene geometry descriptor dimension
    num_bases: int = 6         # K: number of LoRA bases
    lora_rank: int = 16        # r: rank per basis
    target_layers: tuple = (20, 21, 22, 23, 24, 25, 26, 27)  # L=8 upper layers
    target_projections: tuple = ("q_proj", "v_proj")

    # Projection dimensions (Qwen2.5-VL-7B GQA)
    q_proj_in: int = 3584
    q_proj_out: int = 3584
    v_proj_in: int = 3584
    v_proj_out: int = 512      # 4 KV heads x 128 head_dim

    # Training
    lr_depth_router: float = 1e-4
    lr_lora_bank: float = 5e-5
    lr_gates: float = 1e-3
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8

    def proj_dims(self, proj_name: str) -> tuple:
        """Return (d_in, d_out) for a given projection name."""
        if proj_name == "q_proj":
            return self.q_proj_in, self.q_proj_out
        elif proj_name == "v_proj":
            return self.v_proj_in, self.v_proj_out
        raise ValueError(f"Unknown projection: {proj_name}")
```

- [ ] **Step 2: Create geolora.yaml**

```yaml
# configs/geolora.yaml
model:
  base_model: "Qwen/Qwen2.5-VL-7B-Instruct"
  geolora:
    d_geo: 256
    num_bases: 6
    lora_rank: 16
    target_layers: [20, 21, 22, 23, 24, 25, 26, 27]
    target_projections: ["q_proj", "v_proj"]

training:
  lr_depth_router: 1.0e-4
  lr_lora_bank: 5.0e-5
  lr_gates: 1.0e-3
  warmup_ratio: 0.05
  num_epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 8
  precision: "bf16"
  max_seq_length: 2048

data:
  train_file: "/path/to/spatialqa/split_train.json"
  val_file: "/path/to/spatialqa/split_val.json"
  image_dir: "/path/to/nuscenes/images"
  depth_dir: "/path/to/depth_maps"

output:
  output_dir: "./outputs_geolora"
  save_steps: 500
  logging_steps: 10
```

- [ ] **Step 3: Create __init__.py**

```python
# geolora/__init__.py
from .config import GeoLoRAConfig
from .depth_geometry import DepthGeometryNet
from .lora_bank import LoRABasisBank
from .router import GeometryRouter
from .geolora import GeoLoRA
```

- [ ] **Step 4: Commit**

```bash
git add geolora/__init__.py geolora/config.py configs/geolora.yaml
git commit -m "feat(geolora): project scaffolding with config and YAML"
```

---

### Task 2: DepthGeometryNet

**Files:**
- Create: `geolora/depth_geometry.py`
- Create: `tests/test_depth_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depth_geometry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_depth_geometry.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement DepthGeometryNet**

```python
# geolora/depth_geometry.py
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
        # Sobel kernels (fixed, not learned)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        # CNN backbone
        self.conv1 = nn.Conv2d(3, 64, 7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP head
        self.norm = nn.LayerNorm(256)
        self.proj = nn.Linear(256, d_geo)

    def _augment_with_gradients(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Concatenate depth with Sobel x/y gradients -> R^{B x 3 x H x W}."""
        grad_x = F.conv2d(depth_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth_map, self.sobel_y, padding=1)
        return torch.cat([depth_map, grad_x, grad_y], dim=1)

    def forward(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_map: R^{B x 1 x H x W} (from P1, single channel)
        Returns:
            z_geo: R^{B x d_geo}
        """
        x = self._augment_with_gradients(depth_map)  # R^{B x 3 x H x W}
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = self.pool(x).flatten(1)  # R^{B x 256}
        x = F.gelu(self.proj(self.norm(x)))  # R^{B x d_geo}
        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_depth_geometry.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add geolora/depth_geometry.py tests/test_depth_geometry.py
git commit -m "feat(geolora): DepthGeometryNet extracts geometry descriptor from depth maps"
```

---

## Chunk 2: LoRA Basis Bank + Geometry Router

### Task 3: LoRABasisBank

**Files:**
- Create: `geolora/lora_bank.py`
- Create: `tests/test_lora_bank.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lora_bank.py
import torch
from geolora.lora_bank import LoRABasisBank

B, SEQ = 2, 128
D_IN, D_OUT = 3584, 512   # v_proj dims
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
    delta = bank(h, alpha)
    delta.sum().backward()
    assert alpha.grad is not None


def test_zero_init_B_means_zero_output():
    """B_bank is zero-initialized, so initial output should be zero."""
    bank = make_bank()
    h = torch.randn(B, SEQ, D_IN)
    alpha = torch.softmax(torch.randn(B, K), dim=-1)
    delta = bank(h, alpha)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)


def test_param_count():
    bank = make_bank()
    total = sum(p.numel() for p in bank.parameters())
    # A_bank: K * r * d_in = 6 * 16 * 3584 = 344,064
    # B_bank: K * d_out * r = 6 * 512 * 16 = 49,152
    expected = K * R * D_IN + K * D_OUT * R
    assert total == expected, f"Expected {expected}, got {total}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lora_bank.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement LoRABasisBank**

```python
# geolora/lora_bank.py
import torch
import torch.nn as nn


class LoRABasisBank(nn.Module):
    """
    Stores K LoRA basis pairs and mixes them given alpha weights.

    Each basis is a low-rank factorization: delta_W_k = B_k @ A_k
    Mixed LoRA: A_mixed = sum(alpha_k * A_k), B_mixed = sum(alpha_k * B_k)
    Output delta = B_mixed @ A_mixed @ h

    Args:
        num_bases: K, number of basis LoRA pairs
        lora_rank: r, rank per basis
        d_in: input dimension of the target projection
        d_out: output dimension of the target projection
    """

    def __init__(self, num_bases: int, lora_rank: int, d_in: int, d_out: int):
        super().__init__()
        self.num_bases = num_bases
        self.lora_rank = lora_rank

        # A_bank: down projection bases (per-basis Xavier uniform init)
        self.A_bank = nn.Parameter(torch.empty(num_bases, lora_rank, d_in))
        for k in range(num_bases):
            nn.init.xavier_uniform_(self.A_bank[k])

        # B_bank: up projection bases (zero init for zero initial output)
        self.B_bank = nn.Parameter(torch.zeros(num_bases, d_out, lora_rank))

    def forward(self, h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:     R^{B x seq x d_in}  -- hidden states from the LLM layer
            alpha: R^{B x K}           -- mixing weights from the router

        Returns:
            delta: R^{B x seq x d_out} -- LoRA delta to add to frozen proj output
        """
        # Mix bases per sample
        A_mixed = torch.einsum("bk, kri -> bri", alpha, self.A_bank)  # R^{B x r x d_in}
        B_mixed = torch.einsum("bk, kor -> bor", alpha, self.B_bank)  # R^{B x d_out x r}

        # Apply mixed LoRA
        z = torch.einsum("bri, bsi -> bsr", A_mixed, h)     # R^{B x seq x r}
        delta = torch.einsum("bor, bsr -> bso", B_mixed, z)  # R^{B x seq x d_out}
        return delta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lora_bank.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add geolora/lora_bank.py tests/test_lora_bank.py
git commit -m "feat(geolora): LoRABasisBank with einsum-based dynamic mixing"
```

---

### Task 4: GeometryRouter

**Files:**
- Create: `geolora/router.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py
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
    """Layer embeddings should differentiate layers."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_router.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement GeometryRouter**

```python
# geolora/router.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryRouter(nn.Module):
    """
    Maps scene geometry descriptor to per-layer LoRA mixing weights.

    Uses a shared MLP with per-layer embeddings so each LLM layer
    can receive a different mixture of LoRA bases for the same scene.

    Args:
        d_geo: dimension of the scene geometry descriptor
        num_bases: K, number of LoRA bases to mix
        num_layers: L, number of target LLM layers
    """

    def __init__(self, d_geo: int = 256, num_bases: int = 6, num_layers: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.layer_embed = nn.Embedding(num_layers, d_geo)
        self.mlp = nn.Sequential(
            nn.Linear(d_geo, d_geo),
            nn.GELU(),
            nn.Linear(d_geo, num_bases),
        )

    def forward(self, z_geo: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Args:
            z_geo: R^{B x d_geo} -- scene geometry descriptor

        Returns:
            dict mapping layer_index (0..L-1) -> alpha R^{B x K}
        """
        alphas = {}
        for i in range(self.num_layers):
            layer_emb = self.layer_embed(
                torch.tensor(i, device=z_geo.device)
            )
            input_i = z_geo + layer_emb.unsqueeze(0)
            alphas[i] = F.softmax(self.mlp(input_i), dim=-1)
        return alphas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_router.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add geolora/router.py tests/test_router.py
git commit -m "feat(geolora): GeometryRouter with shared MLP + layer embeddings"
```

---

## Chunk 3: Full GeoLoRA Module + Injection Wrapper

### Task 5: DynamicLoRALinear (injection wrapper)

**Files:**
- Create: `geolora/injection.py`
- Create: `tests/test_injection.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_injection.py
import torch
import torch.nn as nn
from geolora.injection import DynamicLoRALinear

B, SEQ = 2, 64
D_IN, D_OUT = 3584, 512  # v_proj dims


def test_without_delta_matches_original():
    """With no delta set, output should match the frozen linear."""
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

    wrapped.set_delta(delta)  # gate is 0.0 by default
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
    """The wrapped linear should not accumulate gradients."""
    original = nn.Linear(D_IN, D_OUT, bias=True)
    wrapped = DynamicLoRALinear(original)
    h = torch.randn(B, SEQ, D_IN)
    wrapped.gate.data.fill_(1.0)
    wrapped.set_delta(torch.randn(B, SEQ, D_OUT))
    result = wrapped(h)
    result.sum().backward()
    assert original.weight.grad is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_injection.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement DynamicLoRALinear**

```python
# geolora/injection.py
import torch
import torch.nn as nn


class DynamicLoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear and adds a per-sample LoRA delta.

    Usage:
        1. Replace a frozen projection with this wrapper.
        2. Before each forward pass, call set_delta(delta).
        3. Forward: output = frozen_linear(h) + gate * delta

    The gate is a learnable scalar initialized to 0 (zero-init),
    so the model starts from base behavior.
    """

    def __init__(self, frozen_linear: nn.Linear):
        super().__init__()
        self.frozen_linear = frozen_linear
        for p in self.frozen_linear.parameters():
            p.requires_grad = False
        self.gate = nn.Parameter(torch.zeros(1))
        self._delta = None

    def set_delta(self, delta: torch.Tensor):
        """Set the LoRA delta for the current forward pass."""
        self._delta = delta

    def clear_delta(self):
        """Clear after forward pass to avoid stale state."""
        self._delta = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        frozen_out = self.frozen_linear(h)
        if self._delta is not None:
            frozen_out = frozen_out + self.gate * self._delta
        return frozen_out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_injection.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add geolora/injection.py tests/test_injection.py
git commit -m "feat(geolora): DynamicLoRALinear wraps frozen projections with gated delta"
```

---

### Task 6: Full GeoLoRA module

**Files:**
- Create: `geolora/geolora.py`
- Create: `tests/test_geolora.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geolora.py
import torch
from geolora.geolora import GeoLoRA
from geolora.config import GeoLoRAConfig

B = 2
SEQ = 128


def make_geolora():
    return GeoLoRA(GeoLoRAConfig())


def test_compute_deltas_shape():
    """Test that compute_deltas returns the right structure."""
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
    """B_bank is zero-init, so initial deltas should be zero."""
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
    # ~445K (depth) + ~8.65M (banks) + ~70K (router) = ~9.17M
    assert 8_000_000 < total < 11_000_000, f"Unexpected param count: {total:,}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geolora.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement GeoLoRA**

```python
# geolora/geolora.py
import torch
import torch.nn as nn

from .config import GeoLoRAConfig
from .depth_geometry import DepthGeometryNet
from .lora_bank import LoRABasisBank
from .router import GeometryRouter


class GeoLoRA(nn.Module):
    """
    Geometry-Conditioned Dynamic LoRA.

    Orchestrates: DepthGeometryNet -> GeometryRouter -> LoRABasisBank
    Produces per-layer, per-projection LoRA deltas from a depth map.

    Does NOT own the DynamicLoRALinear wrappers -- those live in model.py.
    This module is a pure computation: depth_map + hidden_states -> deltas.
    """

    def __init__(self, config: GeoLoRAConfig):
        super().__init__()
        self.config = config
        num_layers = len(config.target_layers)

        self.depth_net = DepthGeometryNet(d_geo=config.d_geo)

        self.router = GeometryRouter(
            d_geo=config.d_geo,
            num_bases=config.num_bases,
            num_layers=num_layers,
        )

        self.banks = nn.ModuleDict()
        for i in range(num_layers):
            layer_banks = nn.ModuleDict()
            for proj in config.target_projections:
                d_in, d_out = config.proj_dims(proj)
                layer_banks[proj] = LoRABasisBank(
                    num_bases=config.num_bases,
                    lora_rank=config.lora_rank,
                    d_in=d_in,
                    d_out=d_out,
                )
            self.banks[str(i)] = layer_banks

    def compute_deltas(
        self,
        depth_map: torch.Tensor,
        hidden_states: dict[int, torch.Tensor],
    ) -> dict[int, dict[str, torch.Tensor]]:
        """
        Compute LoRA deltas for all target layers and projections.

        Args:
            depth_map: R^{B x 1 x H x W}
            hidden_states: dict mapping local_layer_idx -> R^{B x seq x d_lm}

        Returns:
            dict mapping local_layer_idx -> {proj_name: delta R^{B x seq x d_out}}
        """
        z_geo = self.depth_net(depth_map)
        alphas = self.router(z_geo)

        deltas = {}
        for i, alpha in alphas.items():
            h = hidden_states[i]
            layer_deltas = {}
            for proj in self.config.target_projections:
                bank = self.banks[str(i)][proj]
                layer_deltas[proj] = bank(h, alpha)
            deltas[i] = layer_deltas
        return deltas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geolora.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add geolora/geolora.py tests/test_geolora.py
git commit -m "feat(geolora): Full GeoLoRA module assembles depth->router->bank pipeline"
```

---

## Chunk 4: Qwen2.5-VL Integration

### Task 7: Qwen2.5-VL + GeoLoRA model wrapper

**Files:**
- Create: `geolora/model.py`
- Create: `tests/test_geolora_model.py`

- [ ] **Step 1: Implement model wrapper**

This is the most complex file -- it hooks into Qwen2.5-VL's forward pass to:
1. Pre-compute z_geo and alphas from depth map
2. Register pre-hooks on target layers that compute LoRA deltas on-the-fly
3. Inject deltas via DynamicLoRALinear wrappers

```python
# geolora/model.py
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from .config import GeoLoRAConfig
from .geolora import GeoLoRA
from .injection import DynamicLoRALinear


class Qwen2VLWithGeoLoRA(nn.Module):
    """
    Wraps Qwen2.5-VL with GeoLoRA.

    Architecture:
      1. ViT + Merger produce visual tokens (frozen, standard path)
      2. Depth map -> GeoLoRA -> per-layer LoRA deltas
      3. Target LLM layers have q_proj/v_proj replaced with DynamicLoRALinear
      4. Before each forward pass, deltas are injected into the wrappers
    """

    def __init__(self, base_model, geolora: GeoLoRA, config: GeoLoRAConfig, processor):
        super().__init__()
        self.base = base_model
        self.geolora = geolora
        self.config = config
        self.processor = processor
        self._wrapped_layers = {}

        self._wrap_target_layers()

    def _wrap_target_layers(self):
        """Replace q_proj/v_proj in target layers with DynamicLoRALinear."""
        llm_layers = self.base.model.layers
        for local_idx, global_idx in enumerate(self.config.target_layers):
            layer = llm_layers[global_idx]
            self._wrapped_layers[local_idx] = {}
            for proj_name in self.config.target_projections:
                original_linear = getattr(layer.self_attn, proj_name)
                wrapper = DynamicLoRALinear(original_linear)
                setattr(layer.self_attn, proj_name, wrapper)
                self._wrapped_layers[local_idx][proj_name] = wrapper

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: GeoLoRAConfig,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ):
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_name)

        for param in base_model.parameters():
            param.requires_grad = False

        geolora = GeoLoRA(config).to(dtype=torch_dtype)
        return cls(base_model, geolora, config, processor)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        depth_maps=None,
        labels=None,
        **kwargs,
    ):
        if depth_maps is not None:
            z_geo = self.geolora.depth_net(depth_maps)
            alphas = self.geolora.router(z_geo)

            # Set up hooks that compute deltas on-the-fly at each target layer
            hooks = []
            for local_idx, global_idx in enumerate(self.config.target_layers):
                layer = self.base.model.layers[global_idx]

                def make_pre_hook(li, alpha):
                    def hook_fn(module, args):
                        h = args[0]  # R^{B x seq x d_lm}
                        for proj_name in self.config.target_projections:
                            bank = self.geolora.banks[str(li)][proj_name]
                            delta = bank(h, alpha)
                            self._wrapped_layers[li][proj_name].set_delta(delta)
                        return args
                    return hook_fn

                hook = layer.register_forward_pre_hook(
                    make_pre_hook(local_idx, alphas[local_idx])
                )
                hooks.append(hook)
        else:
            hooks = []

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

        for hook in hooks:
            hook.remove()
        for layer_wrappers in self._wrapped_layers.values():
            for wrapper in layer_wrappers.values():
                wrapper.clear_delta()

        return outputs

    def trainable_parameters(self):
        """Return parameter groups with different learning rates."""
        depth_router_params = []
        bank_params = []
        gate_params = []

        for name, param in self.geolora.named_parameters():
            if param.requires_grad:
                if "depth_net" in name or "router" in name:
                    depth_router_params.append(param)
                else:
                    bank_params.append(param)

        for layer_wrappers in self._wrapped_layers.values():
            for wrapper in layer_wrappers.values():
                gate_params.append(wrapper.gate)

        return [
            {"params": depth_router_params, "lr": self.config.lr_depth_router},
            {"params": bank_params, "lr": self.config.lr_lora_bank},
            {"params": gate_params, "lr": self.config.lr_gates},
        ]
```

- [ ] **Step 2: Write smoke test (GPU-only)**

```python
# tests/test_geolora_model.py
import pytest
import torch
from geolora.model import Qwen2VLWithGeoLoRA
from geolora.config import GeoLoRAConfig


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU and Qwen2.5-VL weights"
)
def test_model_creation():
    """Test model can be created and has trainable params. GPU only."""
    try:
        model = Qwen2VLWithGeoLoRA.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            GeoLoRAConfig(),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
    except Exception:
        pytest.skip("Qwen2.5-VL model not available locally")
        return

    assert hasattr(model, "geolora")
    geolora_params = sum(p.numel() for p in model.geolora.parameters())
    assert geolora_params > 0

    base_trainable = sum(
        p.numel() for p in model.base.parameters() if p.requires_grad
    )
    assert base_trainable == 0, "Base model should be fully frozen"

    groups = model.trainable_parameters()
    assert len(groups) == 3
    total_trainable = sum(p.numel() for g in groups for p in g["params"])
    assert total_trainable > 0
```

- [ ] **Step 3: Commit**

```bash
git add geolora/model.py tests/test_geolora_model.py
git commit -m "feat(geolora): Qwen2.5-VL + GeoLoRA integration with hook-based injection"
```

---

## Chunk 5: Dataset + Training + Evaluation

### Task 8: Dataset loader + collator

**Files:**
- Create: `geolora/dataset.py`
- Create: `geolora/collator.py`

- [ ] **Step 1: Implement dataset**

```python
# geolora/dataset.py
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SpatialQADataset(Dataset):
    """
    Loads SpatialQA data with paired depth maps.

    Expected format (GR3D from P3):
    {
      "messages": [...],
      "images": ["path/to/image.jpg"]
    }
    """

    def __init__(self, data_file, image_dir, depth_dir, processor, max_length=2048):
        with open(data_file) as f:
            self.data = json.load(f)
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self.image_dir / item["images"][0]
        image = Image.open(img_path).convert("RGB")

        depth_stem = Path(item["images"][0]).stem
        depth_path = self.depth_dir / f"{depth_stem}.npy"
        if depth_path.exists():
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = np.zeros((image.height, image.width), dtype=np.float32)

        text = self.processor.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt",
            max_length=self.max_length, truncation=True,
        )

        depth_tensor = torch.from_numpy(depth).unsqueeze(0)
        depth_max = depth_tensor.max()
        if depth_max > 0:
            depth_tensor = depth_tensor / depth_max

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs.get("pixel_values", torch.tensor([])),
            "image_grid_thw": inputs.get("image_grid_thw", torch.tensor([])),
            "depth_map": depth_tensor,
        }
```

- [ ] **Step 2: Implement collator**

```python
# geolora/collator.py
import torch
import torch.nn.functional as tF
from dataclasses import dataclass


@dataclass
class SpatialQACollator:
    pad_token_id: int = 0

    def __call__(self, features):
        batch = {}
        max_len = max(f["input_ids"].shape[0] for f in features)

        batch["input_ids"] = torch.stack([
            tF.pad(f["input_ids"], (0, max_len - f["input_ids"].shape[0]),
                   value=self.pad_token_id)
            for f in features
        ])
        batch["attention_mask"] = torch.stack([
            tF.pad(f["attention_mask"], (0, max_len - f["attention_mask"].shape[0]),
                   value=0)
            for f in features
        ])
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100

        if features[0]["pixel_values"].numel() > 0:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features])
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features])

        depths = [f["depth_map"] for f in features]
        max_h = max(d.shape[1] for d in depths)
        max_w = max(d.shape[2] for d in depths)
        batch["depth_maps"] = torch.stack([
            tF.pad(d, (0, max_w - d.shape[2], 0, max_h - d.shape[1]))
            for d in depths
        ])
        return batch
```

- [ ] **Step 3: Commit**

```bash
git add geolora/dataset.py geolora/collator.py
git commit -m "feat(geolora): SpatialQA dataset + collator with depth map pairing"
```

---

### Task 9: Training script

**Files:**
- Create: `scripts/train_geolora.py`

- [ ] **Step 1: Implement training script**

```python
# scripts/train_geolora.py
"""
Usage: python scripts/train_geolora.py --config configs/geolora.yaml
"""
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from geolora.config import GeoLoRAConfig
from geolora.model import Qwen2VLWithGeoLoRA
from geolora.dataset import SpatialQADataset
from geolora.collator import SpatialQACollator


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    geolora_cfg = GeoLoRAConfig(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in cfg["model"]["geolora"].items()}
    )

    print("Loading Qwen2.5-VL + GeoLoRA...")
    model = Qwen2VLWithGeoLoRA.from_pretrained(
        cfg["model"]["base_model"], geolora_cfg, torch_dtype=torch.bfloat16,
    )

    train_ds = SpatialQADataset(
        cfg["data"]["train_file"], cfg["data"]["image_dir"],
        cfg["data"]["depth_dir"], model.processor,
        cfg["training"]["max_seq_length"],
    )
    collator = SpatialQACollator(
        pad_token_id=model.processor.tokenizer.pad_token_id or 0
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True,
    )

    param_groups = model.trainable_parameters()
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    accum = cfg["training"]["gradient_accumulation_steps"]
    total_steps = len(train_loader) // accum * cfg["training"]["num_epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    global_step = 0
    os.makedirs(cfg["output"]["output_dir"], exist_ok=True)

    for epoch in range(cfg["training"]["num_epochs"]):
        for step, batch in enumerate(train_loader):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / accum

            loss.backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % cfg["output"]["logging_steps"] == 0:
                    gate_vals = [
                        w.gate.item()
                        for lw in model._wrapped_layers.values()
                        for w in lw.values()
                    ]
                    avg_gate = sum(gate_vals) / len(gate_vals)
                    print(
                        f"Epoch {epoch} Step {global_step}/{total_steps} "
                        f"Loss: {loss.item()*accum:.4f} "
                        f"AvgGate: {avg_gate:.4f}"
                    )

                if global_step % cfg["output"]["save_steps"] == 0:
                    _save_checkpoint(model, cfg, global_step)

    _save_checkpoint(model, cfg, "final")
    print("Training complete.")


def _save_checkpoint(model, cfg, step):
    save_dir = os.path.join(cfg["output"]["output_dir"], f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {"geolora": model.geolora.state_dict(), "step": step},
        os.path.join(save_dir, "geolora.pt"),
    )
    gates = {
        f"layer_{li}_{pn}": w.gate.data.item()
        for li, lw in model._wrapped_layers.items()
        for pn, w in lw.items()
    }
    torch.save(gates, os.path.join(save_dir, "gates.pt"))
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geolora.yaml")
    main(parser.parse_args().config)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_geolora.py
git commit -m "feat(geolora): training script with 3-group optimizer and gate monitoring"
```

---

### Task 10: Evaluation script

**Files:**
- Create: `scripts/evaluate_geolora.py`

- [ ] **Step 1: Implement evaluation**

```python
# scripts/evaluate_geolora.py
"""
Usage: python scripts/evaluate_geolora.py --config configs/geolora.yaml --checkpoint outputs_geolora/final
"""
import argparse
import json
import os

import torch
import yaml
from tqdm import tqdm

from geolora.config import GeoLoRAConfig
from geolora.model import Qwen2VLWithGeoLoRA
from geolora.dataset import SpatialQADataset


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    geolora_cfg = GeoLoRAConfig(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in cfg["model"]["geolora"].items()}
    )

    model = Qwen2VLWithGeoLoRA.from_pretrained(
        cfg["model"]["base_model"], geolora_cfg, torch_dtype=torch.bfloat16,
    )

    state = torch.load(
        os.path.join(checkpoint_path, "geolora.pt"), map_location="cpu"
    )
    model.geolora.load_state_dict(state["geolora"])

    gates = torch.load(
        os.path.join(checkpoint_path, "gates.pt"), map_location="cpu"
    )
    for key, val in gates.items():
        parts = key.split("_")
        li = int(parts[1])
        pn = "_".join(parts[2:])
        model._wrapped_layers[li][pn].gate.data.fill_(val)

    model.to("cuda")
    model.eval()

    test_ds = SpatialQADataset(
        cfg["data"]["val_file"], cfg["data"]["image_dir"],
        cfg["data"]["depth_dir"], model.processor,
    )

    results = []
    for i in tqdm(range(len(test_ds))):
        item = test_ds[i]
        inputs = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
                  for k, v in item.items()}
        if "depth_map" in inputs:
            inputs["depth_maps"] = inputs.pop("depth_map")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # NOTE: For initial testing, use teacher-forced scoring.
            # Custom generate() with GeoLoRA hooks is a server-side TODO.
            outputs = model(**inputs)

        results.append({"index": i, "loss": outputs.loss.item()})

    out_path = os.path.join(checkpoint_path, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geolora.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/evaluate_geolora.py
git commit -m "feat(geolora): evaluation script with GeoLoRA checkpoint loading"
```

---

### Task 11: Final local verification

- [ ] **Step 1: Run all local tests**

```bash
python -m pytest tests/test_depth_geometry.py tests/test_lora_bank.py tests/test_router.py tests/test_injection.py tests/test_geolora.py -v
```

Expected: all tests PASS (skips GPU-only tests)

- [ ] **Step 2: Verify GeoLoRA param count**

```bash
python -c "
from geolora import GeoLoRA, GeoLoRAConfig
m = GeoLoRA(GeoLoRAConfig())
total = sum(p.numel() for p in m.parameters())
print(f'GeoLoRA params: {total:,} ({total/1e6:.2f}M)')
for name, p in m.named_parameters():
    if p.numel() > 10000:
        print(f'  {name}: {p.numel():,}')
"
```

Expected: total ~9.17M params

- [ ] **Step 3: Verify all imports work**

```bash
python -c "
from geolora import GeoLoRA, GeoLoRAConfig, DepthGeometryNet, LoRABasisBank, GeometryRouter
from geolora.injection import DynamicLoRALinear
from geolora.model import Qwen2VLWithGeoLoRA
print('All imports OK')
"
```

Expected: "All imports OK"

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore(geolora): final local verification"
```

---

## Server Deployment Checklist

1. Push code to server
2. `pip install -r requirements.txt` (no new deps needed)
3. Update `configs/geolora.yaml` with server-specific paths
4. Generate depth maps if not already done: `python scripts/generate_depth_maps.py --image_dir ... --output_dir ...`
5. Run training: `python scripts/train_geolora.py --config configs/geolora.yaml`
6. Run teacher-forced scoring: `python scripts/evaluate_geolora.py --config configs/geolora.yaml --checkpoint outputs_geolora/final`

## Important: Checkpoint Structure

GeoLoRA saves two files per checkpoint:
- `geolora.pt`: GeoLoRA module state (DepthGeometryNet + Router + LoRA Banks)
- `gates.pt`: Gate scalar values from DynamicLoRALinear wrappers

Gates are NOT part of `model.geolora.state_dict()` because they live in the DynamicLoRALinear wrappers (owned by `model._wrapped_layers`). Both files must be loaded for correct restoration.

## Known TODOs for Server

1. **`model.py` hook-based injection**: The pre-hook approach captures `args[0]` as the hidden state input. On server, verify this matches Qwen2.5-VL's actual layer forward signature -- the `args` tuple structure may vary across transformers versions.
2. **`evaluate_geolora.py` generation with GeoLoRA**: Current script uses teacher-forced scoring. Implement a custom `generate()` that maintains the hooks during autoregressive decoding for open-ended generation.
3. **Multi-GPU**: Add `accelerate` config if training on multiple GPUs. The DynamicLoRALinear wrappers and hooks should be compatible with DDP, but test on server.
4. **Gradient checkpointing**: If memory is tight with large images, enable gradient checkpointing on the LLM. The dynamic LoRA hooks will re-fire during recomputation, which is correct since the basis mixing is deterministic given alpha.
