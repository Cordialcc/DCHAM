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
