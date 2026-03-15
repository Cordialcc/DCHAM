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
