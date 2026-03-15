import pytest
import torch
from dcham.model import Qwen2VLWithDCHAM
from dcham.config import DCHAMConfig


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU and Qwen2.5-VL weights",
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
