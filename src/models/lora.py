from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear``.

    ``forward(x) = base(x) + (alpha / rank) * B(A(x))``

    ``A`` is Kaiming-initialised and ``B`` is **zero-initialised**, so the adapter
    is an exact no-op at construction: wrapping a trained Linear leaves its outputs
    unchanged until the adapter learns. The wrapped base Linear is frozen
    (``requires_grad=False``); only ``lora_A`` / ``lora_B`` train.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


def wrap_adaln_with_lora(trunk: nn.Module, rank: int, alpha: float) -> int:
    """Replace each AdaLN block's modulation Linear with a :class:`LoRALinear`.

    Targets ``block.adaLN_mod[-1]`` — the single Linear that maps the (time +
    conditioning) embedding to the per-block scale/shift/gate. Returns the number
    of adapters installed. Idempotent: already-wrapped blocks are skipped.
    """
    count = 0
    for block in trunk.blocks:
        mod = block.adaLN_mod
        linear = mod[-1]
        if isinstance(linear, LoRALinear):
            continue
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"Expected adaLN_mod[-1] to be nn.Linear, got {type(linear)!r}"
            )
        mod[-1] = LoRALinear(linear, rank=rank, alpha=alpha)
        count += 1
    return count


def remap_base_state_dict_for_lora(
    state_dict: dict[str, torch.Tensor], model: nn.Module
) -> dict[str, torch.Tensor]:
    """Remap an unwrapped base checkpoint onto a LoRA-wrapped model.

    Wrapping moves each ``...adaLN_mod.1.{weight,bias}`` under ``.base``
    (``...adaLN_mod.1.base.{weight,bias}``). A base checkpoint saved before
    wrapping uses the un-nested keys, so a naive ``load_state_dict`` would treat
    every AdaLN modulation weight as missing and leave it randomly initialised —
    silently destroying the warm start. This inserts ``.base`` where the wrapped
    model expects it.
    """
    model_keys = set(model.state_dict().keys())
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k in model_keys:
            out[k] = v
            continue
        alt = k.replace(".adaLN_mod.1.", ".adaLN_mod.1.base.")
        out[alt if alt in model_keys else k] = v
    return out
