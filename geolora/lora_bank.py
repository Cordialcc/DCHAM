import torch
import torch.nn as nn


class LoRABasisBank(nn.Module):
    """
    Stores K LoRA basis pairs and mixes them given alpha weights.
    Each basis is a low-rank factorization: delta_W_k = B_k @ A_k
    Output delta = B_mixed @ A_mixed @ h
    """

    def __init__(self, num_bases: int, lora_rank: int, d_in: int, d_out: int):
        super().__init__()
        self.num_bases = num_bases
        self.lora_rank = lora_rank

        self.A_bank = nn.Parameter(torch.empty(num_bases, lora_rank, d_in))
        for k in range(num_bases):
            nn.init.xavier_uniform_(self.A_bank[k])

        self.B_bank = nn.Parameter(torch.zeros(num_bases, d_out, lora_rank))

    def forward(self, h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        A_mixed = torch.einsum("bk, kri -> bri", alpha, self.A_bank)
        B_mixed = torch.einsum("bk, kor -> bor", alpha, self.B_bank)
        z = torch.einsum("bri, bsi -> bsr", A_mixed, h)
        delta = torch.einsum("bor, bsr -> bso", B_mixed, z)
        return delta
