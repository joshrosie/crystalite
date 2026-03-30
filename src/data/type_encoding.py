from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pymatgen.core.periodic_table import Element


@dataclass
class AtomicNumberEncoding:
    """Direct element-channel encoding."""

    vz: int

    def __post_init__(self) -> None:
        self.name = "atomic_number"
        # Channels are elements 1..VZ plus one extra pad/mask bucket.
        self.type_dim = int(self.vz) + 1

    def encode_from_A0(self, a0: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        real_mask = ~pad_mask.bool()
        type_indices = torch.where(real_mask, a0.long() - 1, torch.full_like(a0.long(), self.vz))
        type_oh = F.one_hot(type_indices, num_classes=self.type_dim).to(dtype=torch.float32)
        type_oh = torch.where(real_mask[..., None], type_oh, torch.zeros_like(type_oh))
        return type_oh

    def decode_logits_to_A0(
        self,
        type_logits: torch.Tensor,
        pad_mask: torch.Tensor,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        elem_logits = type_logits[..., : self.vz].clone()
        if allowed_mask is not None:
            allowed = allowed_mask.to(device=elem_logits.device, dtype=torch.bool).reshape(-1)
            if allowed.numel() != self.vz:
                raise ValueError(f"allowed_mask must have length {self.vz}, got {allowed.numel()}")
            if allowed.any():
                elem_logits[..., ~allowed] = -1e9
        atom_idx = elem_logits.argmax(dim=-1) + 1
        atom_idx = torch.where(~pad_mask.bool(), atom_idx, torch.zeros_like(atom_idx))
        return atom_idx.to(dtype=torch.long)


class PeriodicTable2DEncoding:
    """CrystalFlow-style periodic-table row/column encoding."""

    def __init__(self, vz: int) -> None:
        self.vz = int(vz)
        self.name = "periodic_table_2d"
        self.num_row = 13
        self.num_col = 15
        self.type_dim = self.num_row + self.num_col
        self.row_col_to_z = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0],
                [11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0],
                [19, 20, 31, 32, 33, 34, 35, 36, 0, 0, 0, 0, 0, 0, 0],
                [37, 38, 49, 50, 51, 52, 53, 54, 0, 0, 0, 0, 0, 0, 0],
                [55, 56, 81, 82, 83, 84, 85, 86, 0, 0, 0, 0, 0, 0, 0],
                [87, 88, 113, 114, 115, 116, 117, 118, 0, 0, 0, 0, 0, 0, 0],
                [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0, 0, 0, 0, 0],
                [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 0, 0, 0, 0, 0],
                [0, 72, 73, 74, 75, 76, 77, 78, 79, 80, 0, 0, 0, 0, 0],
                [0, 104, 105, 106, 107, 108, 109, 110, 111, 112, 0, 0, 0, 0, 0],
                [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
                [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103],
            ],
            dtype=torch.long,
        )
        max_z = int(self.row_col_to_z.max().item())
        z_to_row = torch.full((max_z + 1,), -1, dtype=torch.long)
        z_to_col = torch.full((max_z + 1,), -1, dtype=torch.long)
        for r in range(self.num_row):
            for c in range(self.num_col):
                z = int(self.row_col_to_z[r, c].item())
                if z > 0:
                    z_to_row[z] = r
                    z_to_col[z] = c
        self.z_to_row = z_to_row
        self.z_to_col = z_to_col
        self.valid_cells = (self.row_col_to_z > 0) & (self.row_col_to_z <= self.vz)
        self.valid_rows = self.valid_cells.any(dim=1)
        if not bool(self.valid_rows.any()):
            raise ValueError(f"No valid periodic-table rows for vz={self.vz}.")

    def encode_from_A0(self, a0: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        real_mask = ~pad_mask.bool()
        safe_a0 = torch.where(real_mask, a0.long(), torch.ones_like(a0.long()))
        z_to_row = self.z_to_row.to(device=safe_a0.device)
        z_to_col = self.z_to_col.to(device=safe_a0.device)
        if real_mask.any():
            z_real = safe_a0[real_mask]
            if (z_real <= 0).any() or (z_real >= z_to_row.numel()).any():
                raise ValueError("Found atomic numbers outside periodic-table map range.")
            row_real = z_to_row[z_real]
            col_real = z_to_col[z_real]
            if (row_real < 0).any() or (col_real < 0).any():
                raise ValueError("Found atomic numbers without periodic-table 2D mapping.")
        rows = z_to_row[safe_a0]
        cols = z_to_col[safe_a0]
        rows = rows.clamp_min(0)
        cols = cols.clamp_min(0)
        row_oh = F.one_hot(rows, num_classes=self.num_row)
        col_oh = F.one_hot(cols, num_classes=self.num_col)
        enc = torch.cat([row_oh, col_oh], dim=-1).to(dtype=torch.float32)
        enc = torch.where(real_mask[..., None], enc, torch.zeros_like(enc))
        return enc

    def _allowed_cells_from_elements(
        self, allowed_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        allowed = allowed_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if allowed.numel() != self.vz:
            raise ValueError(f"allowed_mask must have length {self.vz}, got {allowed.numel()}")
        cell_z = self.row_col_to_z.to(device=device)
        out = torch.zeros_like(cell_z, dtype=torch.bool)
        valid = (cell_z > 0) & (cell_z <= self.vz)
        if bool(allowed.any()):
            out[valid] = allowed[cell_z[valid] - 1]
        return out

    def decode_logits_to_A0(
        self,
        type_logits: torch.Tensor,
        pad_mask: torch.Tensor,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode periodic-table channels with joint pair scoring.

        We score each valid (row, col) cell via:
            score[r, c] = row_logits[r] + col_logits[c]
        then take a masked argmax over valid cells (and allowed cells when provided).
        This intentionally differs from CrystalFlow's row-first decode.
        """
        row_logits = type_logits[..., : self.num_row]
        col_logits = type_logits[..., self.num_row :]
        if col_logits.shape[-1] != self.num_col:
            raise ValueError(
                f"Expected {self.num_col} column logits, got {col_logits.shape[-1]}."
            )
        device = type_logits.device
        valid_cells = self.valid_cells.to(device=device)
        pair_scores = row_logits[..., :, None] + col_logits[..., None, :]
        pair_mask = valid_cells
        if allowed_mask is not None:
            allowed_cells = self._allowed_cells_from_elements(allowed_mask, device=device)
            if bool(allowed_cells.any()):
                pair_mask = valid_cells & allowed_cells

        # If allow-mask produces no candidates, fall back to valid periodic-table cells.
        if not bool(pair_mask.any()):
            pair_mask = valid_cells

        pair_scores = pair_scores.masked_fill(~pair_mask, -1e9)
        flat_idx = pair_scores.reshape(*pair_scores.shape[:-2], -1).argmax(dim=-1)
        row_idx = flat_idx // self.num_col
        col_idx = flat_idx % self.num_col

        chosen_valid = valid_cells[row_idx, col_idx]
        if not bool(chosen_valid.all()):
            raise RuntimeError("Decoded an invalid periodic-table cell.")

        row_col_to_z = self.row_col_to_z.to(device=device)
        atom_idx = row_col_to_z[row_idx, col_idx]
        atom_idx = torch.where(~pad_mask.bool(), atom_idx, torch.zeros_like(atom_idx))
        return atom_idx.to(dtype=torch.long)


# Ground-state electron configurations for Z=1..118.
# Format per entry: (noble_gas_idx, s_count, p_count, d_count, f_count)
# noble_gas_idx: 0=none, 1=He, 2=Ne, 3=Ar, 4=Kr, 5=Xe, 6=Rn
# Anomalous (non-Aufbau) ground states are used where well-established.
_GROUND_STATE_EC: tuple[tuple[int, int, int, int, int], ...] = (
    (0, 1, 0, 0, 0),   #   1  H
    (0, 2, 0, 0, 0),   #   2  He
    (1, 1, 0, 0, 0),   #   3  Li
    (1, 2, 0, 0, 0),   #   4  Be
    (1, 2, 1, 0, 0),   #   5  B
    (1, 2, 2, 0, 0),   #   6  C
    (1, 2, 3, 0, 0),   #   7  N
    (1, 2, 4, 0, 0),   #   8  O
    (1, 2, 5, 0, 0),   #   9  F
    (1, 2, 6, 0, 0),   #  10  Ne
    (2, 1, 0, 0, 0),   #  11  Na
    (2, 2, 0, 0, 0),   #  12  Mg
    (2, 2, 1, 0, 0),   #  13  Al
    (2, 2, 2, 0, 0),   #  14  Si
    (2, 2, 3, 0, 0),   #  15  P
    (2, 2, 4, 0, 0),   #  16  S
    (2, 2, 5, 0, 0),   #  17  Cl
    (2, 2, 6, 0, 0),   #  18  Ar
    (3, 1, 0, 0, 0),   #  19  K
    (3, 2, 0, 0, 0),   #  20  Ca
    (3, 2, 0, 1, 0),   #  21  Sc
    (3, 2, 0, 2, 0),   #  22  Ti
    (3, 2, 0, 3, 0),   #  23  V
    (3, 1, 0, 5, 0),   #  24  Cr  [Ar] 3d⁵ 4s¹
    (3, 2, 0, 5, 0),   #  25  Mn
    (3, 2, 0, 6, 0),   #  26  Fe
    (3, 2, 0, 7, 0),   #  27  Co
    (3, 2, 0, 8, 0),   #  28  Ni
    (3, 1, 0, 10, 0),  #  29  Cu  [Ar] 3d¹⁰ 4s¹
    (3, 2, 0, 10, 0),  #  30  Zn
    (3, 2, 1, 10, 0),  #  31  Ga
    (3, 2, 2, 10, 0),  #  32  Ge
    (3, 2, 3, 10, 0),  #  33  As
    (3, 2, 4, 10, 0),  #  34  Se
    (3, 2, 5, 10, 0),  #  35  Br
    (3, 2, 6, 10, 0),  #  36  Kr
    (4, 1, 0, 0, 0),   #  37  Rb
    (4, 2, 0, 0, 0),   #  38  Sr
    (4, 2, 0, 1, 0),   #  39  Y
    (4, 2, 0, 2, 0),   #  40  Zr
    (4, 1, 0, 4, 0),   #  41  Nb  [Kr] 4d⁴ 5s¹
    (4, 1, 0, 5, 0),   #  42  Mo  [Kr] 4d⁵ 5s¹
    (4, 2, 0, 5, 0),   #  43  Tc
    (4, 1, 0, 7, 0),   #  44  Ru  [Kr] 4d⁷ 5s¹
    (4, 1, 0, 8, 0),   #  45  Rh  [Kr] 4d⁸ 5s¹
    (4, 0, 0, 10, 0),  #  46  Pd  [Kr] 4d¹⁰
    (4, 1, 0, 10, 0),  #  47  Ag  [Kr] 4d¹⁰ 5s¹
    (4, 2, 0, 10, 0),  #  48  Cd
    (4, 2, 1, 10, 0),  #  49  In
    (4, 2, 2, 10, 0),  #  50  Sn
    (4, 2, 3, 10, 0),  #  51  Sb
    (4, 2, 4, 10, 0),  #  52  Te
    (4, 2, 5, 10, 0),  #  53  I
    (4, 2, 6, 10, 0),  #  54  Xe
    (5, 1, 0, 0, 0),   #  55  Cs
    (5, 2, 0, 0, 0),   #  56  Ba
    (5, 2, 0, 1, 0),   #  57  La  [Xe] 5d¹ 6s²
    (5, 2, 0, 1, 1),   #  58  Ce  [Xe] 4f¹ 5d¹ 6s²
    (5, 2, 0, 0, 3),   #  59  Pr  [Xe] 4f³ 6s²
    (5, 2, 0, 0, 4),   #  60  Nd
    (5, 2, 0, 0, 5),   #  61  Pm
    (5, 2, 0, 0, 6),   #  62  Sm
    (5, 2, 0, 0, 7),   #  63  Eu
    (5, 2, 0, 1, 7),   #  64  Gd  [Xe] 4f⁷ 5d¹ 6s²
    (5, 2, 0, 0, 9),   #  65  Tb  [Xe] 4f⁹ 6s²
    (5, 2, 0, 0, 10),  #  66  Dy
    (5, 2, 0, 0, 11),  #  67  Ho
    (5, 2, 0, 0, 12),  #  68  Er
    (5, 2, 0, 0, 13),  #  69  Tm
    (5, 2, 0, 0, 14),  #  70  Yb
    (5, 2, 0, 1, 14),  #  71  Lu
    (5, 2, 0, 2, 14),  #  72  Hf
    (5, 2, 0, 3, 14),  #  73  Ta
    (5, 2, 0, 4, 14),  #  74  W
    (5, 2, 0, 5, 14),  #  75  Re
    (5, 2, 0, 6, 14),  #  76  Os
    (5, 2, 0, 7, 14),  #  77  Ir
    (5, 1, 0, 9, 14),  #  78  Pt  [Xe] 4f¹⁴ 5d⁹ 6s¹
    (5, 1, 0, 10, 14), #  79  Au  [Xe] 4f¹⁴ 5d¹⁰ 6s¹
    (5, 2, 0, 10, 14), #  80  Hg
    (5, 2, 1, 10, 14), #  81  Tl
    (5, 2, 2, 10, 14), #  82  Pb
    (5, 2, 3, 10, 14), #  83  Bi
    (5, 2, 4, 10, 14), #  84  Po
    (5, 2, 5, 10, 14), #  85  At
    (5, 2, 6, 10, 14), #  86  Rn
    (6, 1, 0, 0, 0),   #  87  Fr
    (6, 2, 0, 0, 0),   #  88  Ra
    (6, 2, 0, 1, 0),   #  89  Ac
    (6, 2, 0, 2, 0),   #  90  Th
    (6, 2, 0, 1, 2),   #  91  Pa  [Rn] 5f² 6d¹ 7s²
    (6, 2, 0, 1, 3),   #  92  U   [Rn] 5f³ 6d¹ 7s²
    (6, 2, 0, 1, 4),   #  93  Np  [Rn] 5f⁴ 6d¹ 7s²
    (6, 2, 0, 0, 6),   #  94  Pu
    (6, 2, 0, 0, 7),   #  95  Am
    (6, 2, 0, 1, 7),   #  96  Cm  [Rn] 5f⁷ 6d¹ 7s²
    (6, 2, 0, 0, 9),   #  97  Bk
    (6, 2, 0, 0, 10),  #  98  Cf
    (6, 2, 0, 0, 11),  #  99  Es
    (6, 2, 0, 0, 12),  # 100  Fm
    (6, 2, 0, 0, 13),  # 101  Md
    (6, 2, 0, 0, 14),  # 102  No
    (6, 2, 1, 0, 14),  # 103  Lr  [Rn] 5f¹⁴ 7s² 7p¹ (IUPAC 2021)
    (6, 2, 0, 2, 14),  # 104  Rf
    (6, 2, 0, 3, 14),  # 105  Db
    (6, 2, 0, 4, 14),  # 106  Sg
    (6, 2, 0, 5, 14),  # 107  Bh
    (6, 2, 0, 6, 14),  # 108  Hs
    (6, 2, 0, 7, 14),  # 109  Mt
    (6, 2, 0, 8, 14),  # 110  Ds  (predicted)
    (6, 1, 0, 10, 14), # 111  Rg  (predicted, [Rn] 5f¹⁴ 6d¹⁰ 7s¹)
    (6, 2, 0, 10, 14), # 112  Cn
    (6, 2, 1, 10, 14), # 113  Nh
    (6, 2, 2, 10, 14), # 114  Fl
    (6, 2, 3, 10, 14), # 115  Mc
    (6, 2, 4, 10, 14), # 116  Lv
    (6, 2, 5, 10, 14), # 117  Ts
    (6, 2, 6, 10, 14), # 118  Og
)


_CHEM_V2_PERIOD_DIM = 7
_CHEM_V2_GROUP_DIM = 19  # group 0 reserved for f-block / no standard group
_CHEM_V2_BLOCK_DIM = 4
_CHEM_V2_VALENCE_DIM = 4
_CHEM_V2_BLOCK_TO_INDEX = {"s": 0, "p": 1, "d": 2, "f": 3}
_CHEM_V2_GROUP_SLICES = {
    "period": slice(0, _CHEM_V2_PERIOD_DIM),
    "group": slice(_CHEM_V2_PERIOD_DIM, _CHEM_V2_PERIOD_DIM + _CHEM_V2_GROUP_DIM),
    "block": slice(
        _CHEM_V2_PERIOD_DIM + _CHEM_V2_GROUP_DIM,
        _CHEM_V2_PERIOD_DIM + _CHEM_V2_GROUP_DIM + _CHEM_V2_BLOCK_DIM,
    ),
    "valence": slice(
        _CHEM_V2_PERIOD_DIM + _CHEM_V2_GROUP_DIM + _CHEM_V2_BLOCK_DIM,
        _CHEM_V2_PERIOD_DIM
        + _CHEM_V2_GROUP_DIM
        + _CHEM_V2_BLOCK_DIM
        + _CHEM_V2_VALENCE_DIM,
    ),
}


def _encode_from_z_lookup(
    a0: torch.Tensor,
    pad_mask: torch.Tensor,
    z_to_enc: torch.Tensor,
    max_z: int,
    *,
    name: str,
) -> torch.Tensor:
    real_mask = ~pad_mask.bool()
    z_to_enc = z_to_enc.to(device=a0.device)
    safe_a0 = torch.where(real_mask, a0.long(), torch.zeros_like(a0.long()))
    if real_mask.any():
        z_real = safe_a0[real_mask]
        if (z_real <= 0).any() or (z_real > int(max_z)).any():
            raise ValueError(f"Found atomic numbers outside {name} descriptor range [1, {max_z}].")
    safe_a0 = safe_a0.clamp(0, int(max_z))
    enc = z_to_enc[safe_a0]
    enc = torch.where(real_mask[..., None], enc, torch.zeros_like(enc))
    return enc


def _decode_from_snap_table(
    type_logits: torch.Tensor,
    pad_mask: torch.Tensor,
    allowed_mask: torch.Tensor | None,
    *,
    snap_table: torch.Tensor,
    vz: int,
    max_z: int,
) -> torch.Tensor:
    device = type_logits.device
    scores = type_logits @ snap_table.to(device=device).T

    if allowed_mask is not None:
        allowed = allowed_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if allowed.numel() != int(vz):
            raise ValueError(f"allowed_mask must have length {vz}, got {allowed.numel()}")
        allowed = allowed[: int(max_z)]
        if allowed.any():
            scores = scores.masked_fill(~allowed, -1e9)

    atom_idx = scores.argmax(dim=-1) + 1
    atom_idx = torch.where(~pad_mask.bool(), atom_idx, torch.zeros_like(atom_idx))
    return atom_idx.to(dtype=torch.long)


def _build_chem_v2_descriptor_table(
    vz: int,
) -> tuple[int, torch.Tensor, dict[str, slice]]:
    max_z = min(int(vz), len(_GROUND_STATE_EC))
    if max_z <= 0:
        raise ValueError(f"No chemical descriptors available for vz={vz}.")

    period_rows: list[torch.Tensor] = []
    group_rows: list[torch.Tensor] = []
    block_rows: list[torch.Tensor] = []
    valence_rows: list[torch.Tensor] = []
    for z in range(1, max_z + 1):
        el = Element.from_Z(z)
        block = str(el.block).lower()
        if block not in _CHEM_V2_BLOCK_TO_INDEX:
            raise ValueError(f"Unsupported block '{el.block}' for Z={z}.")
        period = int(el.row)
        if period < 1 or period > _CHEM_V2_PERIOD_DIM:
            raise ValueError(f"Unsupported period {period} for Z={z}.")
        group = 0 if block == "f" else int(el.group or 0)
        if group < 0 or group >= _CHEM_V2_GROUP_DIM:
            raise ValueError(f"Unsupported group {group} for Z={z}.")

        _, s, p, d, f = _GROUND_STATE_EC[z - 1]
        period_rows.append(
            F.one_hot(
                torch.tensor(period - 1, dtype=torch.long),
                num_classes=_CHEM_V2_PERIOD_DIM,
            ).to(dtype=torch.float32)
        )
        group_rows.append(
            F.one_hot(
                torch.tensor(group, dtype=torch.long),
                num_classes=_CHEM_V2_GROUP_DIM,
            ).to(dtype=torch.float32)
        )
        block_rows.append(
            F.one_hot(
                torch.tensor(_CHEM_V2_BLOCK_TO_INDEX[block], dtype=torch.long),
                num_classes=_CHEM_V2_BLOCK_DIM,
            ).to(dtype=torch.float32)
        )
        valence_rows.append(torch.tensor([s / 2, p / 6, d / 10, f / 14], dtype=torch.float32))

    descriptor_table = torch.cat(
        [
            torch.stack(period_rows),
            torch.stack(group_rows),
            torch.stack(block_rows),
            torch.stack(valence_rows),
        ],
        dim=-1,
    )
    group_slices = {
        key: slice(value.start, value.stop) for key, value in _CHEM_V2_GROUP_SLICES.items()
    }
    return max_z, descriptor_table, group_slices


def _standardize_and_group_weight(
    descriptor_table: torch.Tensor, group_slices: dict[str, slice]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = descriptor_table.mean(dim=0)
    std = descriptor_table.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    x_std = (descriptor_table - mean) / std
    x_weighted = x_std.clone()
    for group_slice in group_slices.values():
        group_dim = int(group_slice.stop - group_slice.start)
        x_weighted[:, group_slice] *= float(group_dim) ** -0.5
    return mean, std, x_weighted


def _fit_pca_from_weighted_descriptors(
    x_weighted: torch.Tensor, out_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cov = x_weighted.T @ x_weighted
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0.0)
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :out_dim].contiguous()

    pivot_idx = components.abs().argmax(dim=0)
    signs = components[pivot_idx, torch.arange(components.shape[1])]
    signs = torch.where(signs < 0, -torch.ones_like(signs), torch.ones_like(signs))
    components = components * signs.unsqueeze(0)

    projected = x_weighted @ components
    explained_variance_ratio = eigvals / eigvals.sum().clamp_min(1e-12)
    cumulative_explained_variance = explained_variance_ratio.cumsum(dim=0)
    return components, projected, explained_variance_ratio, cumulative_explained_variance


class ElectronConfigEncoding:
    """Electron configuration encoding.

    Each element is encoded as an 11-dimensional vector:
      - 7 channels: one-hot noble gas base (none, He, Ne, Ar, Kr, Xe, Rn)
      - 4 channels: orbital filling fractions (s/2, p/6, d/10, f/14) in [0, 1]

    Ground-state configurations (actual, not Aufbau) are used for all elements.
    Decoding uses cosine-similarity snap to the precomputed table of valid elements,
    guaranteeing a valid atomic number is always returned.
    """

    def __init__(self, vz: int) -> None:
        self.vz = int(vz)
        self.name = "electron_config"
        self.num_noble = 7   # none, He, Ne, Ar, Kr, Xe, Rn
        self.num_orbital = 4  # s, p, d, f
        self.type_dim = self.num_noble + self.num_orbital  # 11

        max_z = min(self.vz, len(_GROUND_STATE_EC))
        if max_z == 0:
            raise ValueError(f"No electron configurations available for vz={self.vz}.")

        # z_to_enc[z] = 11D encoding for element z; row 0 is zeros (pad token).
        enc_list: list[torch.Tensor] = [torch.zeros(self.type_dim)]
        for z in range(1, max_z + 1):
            noble_idx, s, p, d, f = _GROUND_STATE_EC[z - 1]
            noble_oh = F.one_hot(
                torch.tensor(noble_idx), num_classes=self.num_noble
            ).float()
            orbital = torch.tensor([s / 2, p / 6, d / 10, f / 14], dtype=torch.float32)
            enc_list.append(torch.cat([noble_oh, orbital]))
        self.z_to_enc = torch.stack(enc_list)  # (max_z+1, 11)

        # Precomputed unit-norm table for cosine-snap decoding: shape (max_z, 11).
        valid_encs = self.z_to_enc[1 : max_z + 1]
        norms = valid_encs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.snap_table = valid_encs / norms  # (VZ, 11)

    def encode_from_A0(self, a0: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        real_mask = ~pad_mask.bool()
        z_to_enc = self.z_to_enc.to(device=a0.device)
        safe_a0 = torch.where(real_mask, a0.long(), torch.zeros_like(a0.long()))
        safe_a0 = safe_a0.clamp(0, z_to_enc.shape[0] - 1)
        enc = z_to_enc[safe_a0]
        enc = torch.where(real_mask[..., None], enc, torch.zeros_like(enc))
        return enc

    def decode_logits_to_A0(
        self,
        type_logits: torch.Tensor,
        pad_mask: torch.Tensor,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cosine-similarity snap to nearest valid element encoding.

        Scores each element in the precomputed unit-norm table via dot product
        with the (unnormalized) predicted vector, then takes argmax.  This is
        equivalent to cosine similarity since |pred| is constant across elements.
        """
        device = type_logits.device
        snap_table = self.snap_table.to(device=device)  # (VZ, 11)
        scores = type_logits @ snap_table.T              # (..., VZ)

        if allowed_mask is not None:
            allowed = allowed_mask.to(device=device, dtype=torch.bool).reshape(-1)
            if allowed.numel() != self.vz:
                raise ValueError(
                    f"allowed_mask must have length {self.vz}, got {allowed.numel()}"
                )
            if allowed.any():
                scores = scores.masked_fill(~allowed, -1e9)

        atom_idx = scores.argmax(dim=-1) + 1  # 1-indexed Z
        atom_idx = torch.where(~pad_mask.bool(), atom_idx, torch.zeros_like(atom_idx))
        return atom_idx.to(dtype=torch.long)


class ChemDescriptorRawEncoding:
    """Normalized chemistry descriptor without PCA.

    Descriptor groups:
      - true period one-hot (7)
      - true group one-hot (19 with index 0 reserved for f-block)
      - true block one-hot (s/p/d/f)
      - valence orbital occupancies (s/2, p/6, d/10, f/14)

    The descriptor is standardized, group-weighted, and finally L2-normalized so
    train geometry and cosine-snap decode use the same notion of proximity.
    """

    def __init__(self, vz: int) -> None:
        self.vz = int(vz)
        self.name = "subatomic_tokenizer_raw"
        self.max_z, descriptor_table, self.group_slices = _build_chem_v2_descriptor_table(vz=vz)
        self.descriptor_dim = int(descriptor_table.shape[-1])
        self.type_dim = self.descriptor_dim

        descriptor_mean, descriptor_std, weighted = _standardize_and_group_weight(
            descriptor_table, self.group_slices
        )
        self.descriptor_mean = descriptor_mean
        self.descriptor_std = descriptor_std
        self.weighted_descriptor_table = weighted

        prototype_norms = weighted.norm(dim=-1)
        self.prototype_norms_before_normalization = prototype_norms
        normalized = F.normalize(weighted, p=2, dim=-1)

        self.z_to_enc = torch.cat(
            [torch.zeros((1, self.type_dim), dtype=torch.float32), normalized], dim=0
        )
        self.snap_table = normalized

    def encode_from_A0(self, a0: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return _encode_from_z_lookup(
            a0,
            pad_mask,
            self.z_to_enc,
            self.max_z,
            name=self.name,
        )

    def decode_logits_to_A0(
        self,
        type_logits: torch.Tensor,
        pad_mask: torch.Tensor,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _decode_from_snap_table(
            type_logits,
            pad_mask,
            allowed_mask,
            snap_table=self.snap_table,
            vz=self.vz,
            max_z=self.max_z,
        )


class ChemPCAEncodingV2:
    """PCA-compressed version of the v2 chemistry descriptor."""

    def __init__(self, vz: int, pca_dim: int = 16) -> None:
        self.vz = int(vz)
        self.requested_pca_dim = int(pca_dim)
        if self.requested_pca_dim <= 0:
            raise ValueError(f"pca_dim must be > 0, got {self.requested_pca_dim}.")
        self.name = f"subatomic_tokenizer_pca_{self.requested_pca_dim}"

        self.max_z, descriptor_table, self.group_slices = _build_chem_v2_descriptor_table(vz=vz)
        self.descriptor_dim = int(descriptor_table.shape[-1])
        self.descriptor_mean, self.descriptor_std, weighted = _standardize_and_group_weight(
            descriptor_table, self.group_slices
        )
        self.weighted_descriptor_table = weighted
        self.type_dim = min(
            self.requested_pca_dim,
            int(weighted.shape[0]),
            int(weighted.shape[1]),
        )
        if self.type_dim <= 0:
            raise ValueError(
                "chem_pca_v2 requires at least one PCA component; "
                f"got weighted descriptor shape {tuple(weighted.shape)}."
            )

        (
            self.pca_components,
            projected,
            self.explained_variance_ratio,
            self.cumulative_explained_variance,
        ) = _fit_pca_from_weighted_descriptors(weighted, out_dim=self.type_dim)
        prototype_norms = projected.norm(dim=-1)
        self.prototype_norms_before_normalization = prototype_norms
        normalized = F.normalize(projected, p=2, dim=-1)

        self.z_to_enc = torch.cat(
            [torch.zeros((1, self.type_dim), dtype=torch.float32), normalized], dim=0
        )
        self.snap_table = normalized

    def encode_from_A0(self, a0: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return _encode_from_z_lookup(
            a0,
            pad_mask,
            self.z_to_enc,
            self.max_z,
            name=self.name,
        )

    def decode_logits_to_A0(
        self,
        type_logits: torch.Tensor,
        pad_mask: torch.Tensor,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _decode_from_snap_table(
            type_logits,
            pad_mask,
            allowed_mask,
            snap_table=self.snap_table,
            vz=self.vz,
            max_z=self.max_z,
        )


TypeEncoding = (
    AtomicNumberEncoding
    | PeriodicTable2DEncoding
    | ElectronConfigEncoding
    | ChemDescriptorRawEncoding
    | ChemPCAEncodingV2
)


def build_type_encoding(mode: str, vz: int) -> TypeEncoding:
    mode_norm = mode.strip().lower()
    if mode_norm == "atomic_number":
        return AtomicNumberEncoding(vz=vz)
    if mode_norm in {"subatomic_tokenizer_raw", "chem_raw_v2"}:
        return ChemDescriptorRawEncoding(vz=vz)
    if mode_norm in {"subatomic_tokenizer_pca", "chem_pca_v2"}:
        return ChemPCAEncodingV2(vz=vz, pca_dim=24)
    if mode_norm.startswith("subatomic_tokenizer_pca_"):
        suffix = mode_norm.rsplit("_", 1)[-1]
        try:
            pca_dim = int(suffix)
        except ValueError as exc:
            raise ValueError(f"Invalid subatomic_tokenizer_pca mode '{mode}'.") from exc
        return ChemPCAEncodingV2(vz=vz, pca_dim=pca_dim)
    if mode_norm.startswith("chem_pca_v2_"):
        suffix = mode_norm.rsplit("_", 1)[-1]
        try:
            pca_dim = int(suffix)
        except ValueError as exc:
            raise ValueError(f"Invalid chem_pca_v2 mode '{mode}'.") from exc
        return ChemPCAEncodingV2(vz=vz, pca_dim=pca_dim)
    raise ValueError(
        "Unknown type encoding "
        f"'{mode}'. Expected one of: atomic_number, subatomic_tokenizer_raw, "
        "subatomic_tokenizer_pca_8, subatomic_tokenizer_pca_16, subatomic_tokenizer_pca_24."
    )
