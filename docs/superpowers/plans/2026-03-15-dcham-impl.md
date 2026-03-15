# DCHAM Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Depth-Conditioned HyperAttention Module (DCHAM) and integrate it with Qwen2.5-VL-7B for spatial reasoning training and evaluation.

**Architecture:** DCHAM is a standalone module that takes ViT visual tokens + P1 depth maps + text embeddings, produces 16 spatial tokens via depth-conditioned HyperAttention, and concatenates them with visual tokens before the LLM. Base model frozen, DCHAM + LoRA trained.

**Tech Stack:** PyTorch, HuggingFace transformers, PEFT (LoRA), Qwen2.5-VL-7B

**Spec:** `docs/superpowers/specs/2026-03-15-dcham-design.md`

---

## File Structure

```
/Users/daijidong/Documents/zigzagging/
├── docs/                              # existing
├── dcham/
│   ├── __init__.py                    # Package exports
│   ├── config.py                      # All hyperparameters
│   ├── depth_features.py              # DepthFeatureNet (77K params)
│   ├── hyper_attention.py             # BasisKernelLibrary + HyperAttentionHead (874K x 2)
│   ├── module.py                      # Full DCHAM module (6M params)
│   ├── model.py                       # Qwen2.5-VL + DCHAM integration
│   ├── dataset.py                     # SpatialQA dataset with depth maps
│   └── collator.py                    # Data collator
├── scripts/
│   ├── train.py                       # Training entry point
│   ├── evaluate.py                    # Evaluation
│   └── generate_depth_maps.py         # Batch generate depth maps with DAv2
├── configs/
│   └── default.yaml                   # Training config (paths, hyperparams)
├── tests/
│   ├── test_depth_features.py         # DepthFeatureNet dimension tests
│   ├── test_hyper_attention.py        # HyperAttentionHead dimension + gradient tests
│   ├── test_dcham.py                  # Full module integration test
│   └── test_model.py                  # Qwen2.5-VL + DCHAM forward pass test
└── requirements.txt
```

**File responsibilities:**
- `config.py`: Single source of truth for all dimensions (d_vit, d_head, n, r, etc.)
- `depth_features.py`: CNN that processes raw depth maps -> multi-scale features
- `hyper_attention.py`: The core innovation -- basis kernel library + dynamic assembly + tri-modal attention
- `module.py`: Assembles DepthFeatureNet + 2 HyperAttentionHeads + spatial queries + output projection
- `model.py`: Wraps Qwen2.5-VL, hooks into ViT to capture pre-merger tokens, injects spatial tokens
- `dataset.py` + `collator.py`: Loads SpatialQA data with paired depth maps
- `scripts/train.py`: Training loop with warmup schedule (DCHAM-only then DCHAM+LoRA)

---

## Chunk 1: Project Setup + Core Components

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `dcham/__init__.py`
- Create: `dcham/config.py`
- Create: `configs/default.yaml`

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.1.0
transformers>=4.45.0
peft>=0.13.0
accelerate>=0.34.0
pillow
pyyaml
numpy
```

- [ ] **Step 2: Create config.py with all hyperparameters**

```python
# dcham/config.py
from dataclasses import dataclass

@dataclass
class DCHAMConfig:
    # Qwen2.5-VL-7B dimensions (frozen, reference only)
    d_vit: int = 1280
    d_lm: int = 3584
    vit_patch_size: int = 14

    # DCHAM internal dimensions
    d_model: int = 512      # H * d_head
    d_head: int = 64
    num_heads: int = 8
    num_basis: int = 12     # n: basis kernels
    rank: int = 24          # r: rank per basis
    num_queries: int = 16   # M: spatial query tokens (8 fine + 8 coarse)
    d_depth: int = 128      # depth feature dimension

    # Training
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate_dcham: float = 1e-4
    learning_rate_lora: float = 2e-5
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
```

- [ ] **Step 3: Create default.yaml**

```yaml
# configs/default.yaml
model:
  base_model: "Qwen/Qwen2.5-VL-7B-Instruct"
  dcham:
    d_model: 512
    d_head: 64
    num_heads: 8
    num_basis: 12
    rank: 24
    num_queries: 16
    d_depth: 128
  lora:
    rank: 64
    alpha: 128
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  learning_rate_dcham: 1.0e-4
  learning_rate_lora: 2.0e-5
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
  output_dir: "./outputs"
  save_steps: 500
  logging_steps: 10
```

- [ ] **Step 4: Create __init__.py**

```python
# dcham/__init__.py
from .config import DCHAMConfig
from .depth_features import DepthFeatureNet
from .hyper_attention import HyperAttentionHead
from .module import DCHAM
```

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt dcham/__init__.py dcham/config.py configs/default.yaml
git commit -m "feat: project scaffolding with config and dependencies"
```

---

### Task 2: DepthFeatureNet

**Files:**
- Create: `dcham/depth_features.py`
- Create: `tests/test_depth_features.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_depth_features.py
import torch
from dcham.depth_features import DepthFeatureNet

def test_output_shapes():
    net = DepthFeatureNet(d_depth=128)
    depth_map = torch.randn(1, 1, 448, 672)  # typical NuScenes front cam

    # ViT grid for this image: 448/14=32, 672/14=48
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_depth_features.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement DepthFeatureNet**

```python
# dcham/depth_features.py
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
        x = F.gelu(self.conv2(x))  # R^{B x d_depth x h' x w'}

        fine = F.adaptive_avg_pool2d(x, (vit_grid_h, vit_grid_w))
        fine = fine.flatten(2).transpose(1, 2)  # R^{B x N x d_depth}

        coarse = self.pool_coarse(x)
        coarse = coarse.flatten(2).transpose(1, 2)  # R^{B x 16 x d_depth}

        glob = self.pool_global(x).flatten(1)  # R^{B x d_depth}

        return {"fine": fine, "coarse": coarse, "global": glob}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_depth_features.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add dcham/depth_features.py tests/test_depth_features.py
git commit -m "feat: DepthFeatureNet extracts multi-scale features from depth maps"
```

---

### Task 3: HyperAttentionHead (core innovation)

**Files:**
- Create: `dcham/hyper_attention.py`
- Create: `tests/test_hyper_attention.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_hyper_attention.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hyper_attention.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement HyperAttentionHead**

```python
# dcham/hyper_attention.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperAttentionHead(nn.Module):
    """
    Core innovation: depth features dynamically generate attention projection
    matrices via basis kernel combination.

    Depth -> HyperNetwork -> alpha coefficients -> W_K, W_V from basis library
    Then: tri-modal attention with Q (text), K/V (depth-generated from RGB)
    """

    def __init__(
        self,
        d_vit: int = 1280,
        d_head: int = 64,
        num_heads: int = 8,
        num_basis: int = 12,
        rank: int = 24,
        d_depth: int = 128,
    ):
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads
        self.num_basis = num_basis
        self.scale = math.sqrt(d_head)

        # Context Encoder: global depth -> context vector
        ctx_dim = d_depth * 2
        self.context_enc = nn.Sequential(
            nn.Linear(d_depth, d_depth),
            nn.GELU(),
            nn.LayerNorm(d_depth),
            nn.Linear(d_depth, ctx_dim),
            nn.GELU(),
            nn.LayerNorm(ctx_dim),
        )

        # Coefficient Generator: context -> per-head basis weights
        self.coeff_K = nn.Linear(ctx_dim, num_heads * num_basis)
        self.coeff_V = nn.Linear(ctx_dim, num_heads * num_basis)

        # Basis Kernel Library (core learnable parameters)
        # Each basis: B_i = U_i @ V_i^T  in R^{d_head x d_vit}
        self.U_K = nn.Parameter(torch.randn(num_basis, d_head, rank) * 0.02)
        self.V_K = nn.Parameter(torch.randn(num_basis, d_vit, rank) * 0.02)
        self.U_V = nn.Parameter(torch.randn(num_basis, d_head, rank) * 0.02)
        self.V_V = nn.Parameter(torch.randn(num_basis, d_vit, rank) * 0.02)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        depth_global: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: R^{B x N x d_vit} -- ViT output (frozen)
            depth_global:  R^{B x d_depth}   -- global depth feature
            Q:             R^{B x M x H x d_head} -- spatial queries

        Returns:
            R^{B x M x (H * d_head)} -- spatial token features
        """
        B = visual_tokens.shape[0]

        # --- HyperNetwork: depth -> basis coefficients ---
        ctx = self.context_enc(depth_global)  # R^{B x ctx_dim}
        alpha_K = self.coeff_K(ctx).view(B, self.num_heads, self.num_basis)
        alpha_V = self.coeff_V(ctx).view(B, self.num_heads, self.num_basis)
        alpha_K = F.softmax(alpha_K, dim=-1)  # R^{B x H x n}
        alpha_V = F.softmax(alpha_V, dim=-1)

        # --- Basis kernel assembly ---
        K = self._assemble(visual_tokens, alpha_K, self.V_K, self.U_K)
        V = self._assemble(visual_tokens, alpha_V, self.V_V, self.U_V)
        # K, V: R^{B x N x H x d_head}

        # --- Tri-modal attention ---
        attn = torch.einsum("bmhd, bnhd -> bhmn", Q, K) / self.scale
        attn = F.softmax(attn, dim=-1)  # R^{B x H x M x N}
        out = torch.einsum("bhmn, bnhd -> bmhd", attn, V)

        return out.reshape(B, Q.shape[1], self.num_heads * self.d_head)

    def _assemble(
        self,
        X: torch.Tensor,
        alpha: torch.Tensor,
        V_basis: nn.Parameter,
        U_basis: nn.Parameter,
    ) -> torch.Tensor:
        """
        Efficient basis kernel assembly. Never materializes full W matrix.

        X:       R^{B x N x d_vit}
        alpha:   R^{B x H x n}
        V_basis: R^{n x d_vit x r}
        U_basis: R^{n x d_head x r}

        Returns: R^{B x N x H x d_head}
        """
        # Step 1: project to rank space per basis
        Z = torch.einsum("bnd, kdr -> bnkr", X, V_basis)  # R^{B x N x n x r}
        # Step 2: project to head space per basis
        K_all = torch.einsum("bnkr, kfr -> bnkf", Z, U_basis)  # R^{B x N x n x d_head}
        # Step 3: weighted combination
        out = torch.einsum("bhk, bnkf -> bnhf", alpha, K_all)  # R^{B x N x H x d_head}
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hyper_attention.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add dcham/hyper_attention.py tests/test_hyper_attention.py
git commit -m "feat: HyperAttentionHead with depth-conditioned basis kernel attention"
```

---

## Chunk 2: Full DCHAM Module + Qwen2.5-VL Integration

### Task 4: Full DCHAM module

**Files:**
- Create: `dcham/module.py`
- Create: `tests/test_dcham.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_dcham.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dcham.py -v`
Expected: FAIL

- [ ] **Step 3: Implement DCHAM module**

```python
# dcham/module.py
import torch
import torch.nn as nn

from .config import DCHAMConfig
from .depth_features import DepthFeatureNet
from .hyper_attention import HyperAttentionHead


class DCHAM(nn.Module):
    """
    Depth-Conditioned HyperAttention Module.

    Components:
      1. DepthFeatureNet: depth map -> multi-scale spatial features
      2. HyperAttentionHead x 2 (fine + coarse): depth generates W_K, W_V
      3. Spatial queries + text conditioning: defines extraction targets
      4. Output projection: projects to LLM dimension
    """

    def __init__(self, config: DCHAMConfig):
        super().__init__()
        self.config = config
        M = config.num_queries
        d_model = config.d_model
        d_lm = config.d_lm

        # Component 1: Depth Feature Network
        self.depth_net = DepthFeatureNet(d_depth=config.d_depth)

        # Component 2: Two-scale HyperAttention
        head_kwargs = dict(
            d_vit=config.d_vit, d_head=config.d_head,
            num_heads=config.num_heads, num_basis=config.num_basis,
            rank=config.rank, d_depth=config.d_depth,
        )
        self.hyper_fine = HyperAttentionHead(**head_kwargs)
        self.hyper_coarse = HyperAttentionHead(**head_kwargs)

        # Component 3: Spatial queries + text conditioning
        self.queries = nn.Parameter(torch.randn(M, d_model) * 0.02)
        self.text_proj = nn.Sequential(
            nn.Linear(d_lm, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.head_proj = nn.Linear(d_model, config.num_heads * config.d_head)

        # Component 4: Output
        self.scale_embed = nn.Embedding(2, d_model)
        self.out_proj = nn.Linear(d_model, d_lm)
        self.out_norm = nn.LayerNorm(d_lm)

    def forward(
        self, visual_tokens: torch.Tensor, depth_map: torch.Tensor,
        text_embeddings: torch.Tensor, vit_grid_h: int, vit_grid_w: int,
    ) -> torch.Tensor:
        B = visual_tokens.shape[0]
        cfg = self.config
        M_half = cfg.num_queries // 2

        # 1. Depth features
        depth_feats = self.depth_net(depth_map, vit_grid_h, vit_grid_w)

        # 2. Text-conditioned spatial queries
        text_pool = text_embeddings.mean(dim=1)
        text_mod = self.text_proj(text_pool)
        Q = self.queries.unsqueeze(0) + text_mod.unsqueeze(1)
        Q = self.head_proj(Q).view(B, cfg.num_queries, cfg.num_heads, cfg.d_head)
        Q_fine, Q_coarse = Q[:, :M_half], Q[:, M_half:]

        # 3. Two-scale HyperAttention
        S_fine = self.hyper_fine(visual_tokens, depth_feats["global"], Q_fine)
        S_coarse = self.hyper_coarse(visual_tokens, depth_feats["global"], Q_coarse)

        # 4. Scale embedding + concatenate
        dev = S_fine.device
        S_fine = S_fine + self.scale_embed(torch.zeros(1, dtype=torch.long, device=dev))
        S_coarse = S_coarse + self.scale_embed(torch.ones(1, dtype=torch.long, device=dev))
        S = torch.cat([S_fine, S_coarse], dim=1)

        # 5. Project to LLM dimension
        return self.out_norm(self.out_proj(S))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_dcham.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add dcham/module.py tests/test_dcham.py
git commit -m "feat: Full DCHAM module assembles all components"
```

---

### Task 5: Qwen2.5-VL + DCHAM integration

**Files:**
- Create: `dcham/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Implement model wrapper**

```python
# dcham/model.py
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

from .config import DCHAMConfig
from .module import DCHAM


class Qwen2VLWithDCHAM(nn.Module):
    """
    Wraps Qwen2.5-VL with DCHAM module.

    Architecture:
      1. ViT processes image -> visual_tokens (pre-merger captured via hook)
      2. Merger produces merged_visual_tokens (standard path)
      3. DCHAM produces spatial_tokens from pre-merger tokens + depth
      4. spatial_tokens are injected into the LLM input sequence
    """

    def __init__(self, base_model, dcham: DCHAM, processor):
        super().__init__()
        self.base = base_model
        self.dcham = dcham
        self.processor = processor
        self._pre_merger_features = None

        # Hook to capture ViT output BEFORE merger
        self._hook = self.base.visual.merger.register_forward_pre_hook(
            self._capture_pre_merger
        )

    def _capture_pre_merger(self, module, args):
        self._pre_merger_features = args[0].detach().clone()

    @classmethod
    def from_pretrained(
        cls, model_name: str, dcham_config: DCHAMConfig,
        torch_dtype=torch.bfloat16, device_map="auto",
    ):
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_name)

        # Freeze ViT + Merger
        for param in base_model.visual.parameters():
            param.requires_grad = False

        # Apply LoRA to LLM
        lora_config = LoraConfig(
            r=dcham_config.lora_rank, lora_alpha=dcham_config.lora_alpha,
            target_modules=list(dcham_config.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_config)

        dcham = DCHAM(dcham_config).to(dtype=torch_dtype)
        return cls(base_model, dcham, processor)

    def forward(
        self, input_ids, attention_mask, pixel_values, image_grid_thw,
        depth_maps, labels=None,
    ):
        # Step 1: Run visual encoder (hook captures pre-merger features)
        image_embeds = self.base.visual(pixel_values, grid_thw=image_grid_thw)
        pre_merger = self._pre_merger_features
        self._pre_merger_features = None

        # Step 2: Get text embeddings for DCHAM conditioning
        embed_layer = self.base.get_input_embeddings()
        text_embeds = embed_layer(input_ids)

        # Step 3: Run DCHAM per batch item
        B = input_ids.shape[0]
        spatial_tokens_list = []
        offset = 0
        for i in range(B):
            t, h, w = image_grid_thw[i].tolist()
            n_patches = int(t * h * w)
            vis_tok = pre_merger[offset:offset + n_patches].unsqueeze(0)
            offset += n_patches
            st = self.dcham(
                vis_tok, depth_maps[i:i+1], text_embeds[i:i+1],
                int(h), int(w),
            )
            spatial_tokens_list.append(st.squeeze(0))

        # Step 4: Build inputs_embeds with visual + spatial tokens
        # NOTE: This is a simplified version. On the server, refine to match
        # Qwen2.5-VL's exact <|vision_start|>/<|vision_end|> token protocol.
        inputs_embeds = self._build_inputs_embeds(
            input_ids, text_embeds, image_embeds,
            spatial_tokens_list, image_grid_thw,
        )

        # Extend attention mask for spatial tokens
        M = self.dcham.config.num_queries
        spatial_mask = torch.ones(B, M, device=attention_mask.device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([spatial_mask, attention_mask], dim=1)

        # Step 5: Forward through LLM
        # Access the inner model through PEFT wrapper
        outputs = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=self._extend_labels(labels, M) if labels is not None else None,
        )
        return outputs

    def _build_inputs_embeds(
        self, input_ids, text_embeds, image_embeds,
        spatial_tokens_list, image_grid_thw,
    ):
        """Prepend spatial tokens to the input sequence."""
        B = input_ids.shape[0]
        device = text_embeds.device
        dtype = text_embeds.dtype

        # Let the base model handle visual token merging into text
        # We prepend spatial tokens before the full sequence
        # The base model's forward already merges vision+text
        merged_list = []
        for i in range(B):
            st = spatial_tokens_list[i]  # R^{M x d_lm}
            merged_list.append(torch.cat([st, text_embeds[i]], dim=0))

        max_len = max(m.shape[0] for m in merged_list)
        padded = torch.zeros(B, max_len, text_embeds.shape[-1], device=device, dtype=dtype)
        for i, m in enumerate(merged_list):
            padded[i, :m.shape[0]] = m

        return padded

    def _extend_labels(self, labels, M):
        """Prepend -100 (ignore) for spatial token positions."""
        if labels is None:
            return None
        B = labels.shape[0]
        ignore = torch.full((B, M), -100, device=labels.device, dtype=labels.dtype)
        return torch.cat([ignore, labels], dim=1)
```

- [ ] **Step 2: Write smoke test**

```python
# tests/test_model.py
import pytest
import torch
from dcham.model import Qwen2VLWithDCHAM
from dcham.config import DCHAMConfig

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU and Qwen2.5-VL weights"
)
def test_model_creation():
    """Test model can be created. Run on server only."""
    try:
        model = Qwen2VLWithDCHAM.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            DCHAMConfig(),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
    except Exception:
        pytest.skip("Qwen2.5-VL model not available locally")
        return
    assert hasattr(model, "dcham")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0
```

- [ ] **Step 3: Commit**

```bash
git add dcham/model.py tests/test_model.py
git commit -m "feat: Qwen2.5-VL + DCHAM integration wrapper"
```

---

## Chunk 3: Dataset + Training + Evaluation

### Task 6: Dataset loader + collator

**Files:**
- Create: `dcham/dataset.py`
- Create: `dcham/collator.py`

- [ ] **Step 1: Implement dataset**

```python
# dcham/dataset.py
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
      "messages": [
        {"role": "user", "content": "<spatial_list>...<image>Question"},
        {"role": "assistant", "content": "Answer"}
      ],
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
# dcham/collator.py
import torch
import torch.nn.functional as tF
from dataclasses import dataclass


@dataclass
class SpatialQACollator:
    pad_token_id: int = 0

    def __call__(self, features: list[dict]) -> dict:
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
git add dcham/dataset.py dcham/collator.py
git commit -m "feat: SpatialQA dataset loader with depth map pairing"
```

---

### Task 7: Training script

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: Implement training script**

```python
# scripts/train.py
"""
Usage: python scripts/train.py --config configs/default.yaml
"""
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from dcham.config import DCHAMConfig
from dcham.model import Qwen2VLWithDCHAM
from dcham.dataset import SpatialQADataset
from dcham.collator import SpatialQACollator


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dcham_cfg = DCHAMConfig(**cfg["model"]["dcham"])
    dcham_cfg.lora_rank = cfg["model"]["lora"]["rank"]
    dcham_cfg.lora_alpha = cfg["model"]["lora"]["alpha"]

    print("Loading Qwen2.5-VL + DCHAM...")
    model = Qwen2VLWithDCHAM.from_pretrained(
        cfg["model"]["base_model"], dcham_cfg, torch_dtype=torch.bfloat16,
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

    dcham_params = list(model.dcham.parameters())
    lora_params = [p for n, p in model.base.named_parameters() if p.requires_grad and "lora" in n]
    optimizer = torch.optim.AdamW([
        {"params": dcham_params, "lr": cfg["training"]["learning_rate_dcham"]},
        {"params": lora_params, "lr": cfg["training"]["learning_rate_lora"]},
    ], weight_decay=0.01)

    accum = cfg["training"]["gradient_accumulation_steps"]
    total_steps = len(train_loader) // accum * cfg["training"]["num_epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    global_step = 0
    os.makedirs(cfg["output"]["output_dir"], exist_ok=True)

    for epoch in range(cfg["training"]["num_epochs"]):
        for step, batch in enumerate(train_loader):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / accum

            loss.backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % cfg["output"]["logging_steps"] == 0:
                    print(f"Epoch {epoch} Step {global_step}/{total_steps} Loss: {loss.item()*accum:.4f}")

                if global_step % cfg["output"]["save_steps"] == 0:
                    save_dir = f"{cfg['output']['output_dir']}/checkpoint-{global_step}"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({"dcham": model.dcham.state_dict(), "step": global_step},
                               f"{save_dir}/dcham.pt")
                    model.base.save_pretrained(save_dir)
                    print(f"Saved to {save_dir}")

    # Save final
    save_dir = f"{cfg['output']['output_dir']}/final"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"dcham": model.dcham.state_dict()}, f"{save_dir}/dcham.pt")
    model.base.save_pretrained(save_dir)
    print(f"Training complete. Final model at {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train.py
git commit -m "feat: training script with dual-LR optimizer and warmup"
```

---

### Task 8: Evaluation + depth map generation scripts

**Files:**
- Create: `scripts/evaluate.py`
- Create: `scripts/generate_depth_maps.py`

- [ ] **Step 1: Implement evaluation**

```python
# scripts/evaluate.py
"""
Usage: python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/final
"""
import argparse
import json

import torch
import yaml
from tqdm import tqdm

from dcham.config import DCHAMConfig
from dcham.model import Qwen2VLWithDCHAM
from dcham.dataset import SpatialQADataset


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dcham_cfg = DCHAMConfig(**cfg["model"]["dcham"])
    dcham_cfg.lora_rank = cfg["model"]["lora"]["rank"]
    dcham_cfg.lora_alpha = cfg["model"]["lora"]["alpha"]

    model = Qwen2VLWithDCHAM.from_pretrained(
        cfg["model"]["base_model"], dcham_cfg, torch_dtype=torch.bfloat16,
    )
    state = torch.load(f"{checkpoint_path}/dcham.pt", map_location="cpu")
    model.dcham.load_state_dict(state["dcham"])
    model.base.load_adapter(checkpoint_path)
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
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = model.base.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                max_new_tokens=256,
            )
        pred = model.processor.decode(gen[0], skip_special_tokens=True)
        results.append({"index": i, "prediction": pred})

    out_path = f"{checkpoint_path}/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
```

- [ ] **Step 2: Implement depth map generation**

```python
# scripts/generate_depth_maps.py
"""
Usage: python scripts/generate_depth_maps.py --image_dir /path/to/images --output_dir /path/to/depth
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


def main(image_dir: str, output_dir: str, model_size: str = "vits"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "depth-estimation",
        model=f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf",
        device=device,
    )
    image_files = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))

    for img_path in tqdm(image_files, desc="Generating depth maps"):
        out_path = Path(output_dir) / f"{img_path.stem}.npy"
        if out_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        result = pipe(image)
        depth = np.array(result["depth"], dtype=np.float32)
        np.save(out_path, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_size", default="vits", choices=["vits", "vitb", "vitl"])
    args = parser.parse_args()
    main(args.image_dir, args.output_dir, args.model_size)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py scripts/generate_depth_maps.py
git commit -m "feat: evaluation and depth map generation scripts"
```

---

### Task 9: Final verification

- [ ] **Step 1: Run all local tests**

```bash
python -m pytest tests/ -v --ignore=tests/test_model.py
```

Expected: all tests PASS

- [ ] **Step 2: Verify DCHAM param count**

```bash
python -c "
from dcham import DCHAM, DCHAMConfig
m = DCHAM(DCHAMConfig())
total = sum(p.numel() for p in m.parameters())
print(f'DCHAM params: {total:,} ({total/1e6:.1f}M)')
for name, p in m.named_parameters():
    print(f'  {name}: {p.numel():,}')
"
```

Expected: total ~6M params

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final local verification"
```

---

## Server Deployment Checklist

1. Push code to server
2. `pip install -r requirements.txt`
3. Update `configs/default.yaml` with server-specific paths
4. Generate depth maps: `python scripts/generate_depth_maps.py --image_dir ... --output_dir ...`
5. Run training: `python scripts/train.py --config configs/default.yaml`
6. Evaluate: `python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/final`

## Known TODOs for Server

1. **`model.py` visual token merging**: Current `_build_inputs_embeds` is simplified (prepends spatial tokens). On server, test with actual Qwen2.5-VL and refine to properly handle `<|vision_start|>` / `<|vision_end|>` markers.
2. **Multi-GPU**: Add `accelerate` config if needed.
3. **Warmup phase**: Optionally freeze LoRA during first 5% steps (train DCHAM only first).
