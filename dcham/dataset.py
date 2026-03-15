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
