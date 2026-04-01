from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pymatgen.core.periodic_table import Element


_SUPPORTED_TYPE_ENCODING_NAMES = (
    "atomic_number",
    "subatomic_tokenizer_raw",
    "subatomic_tokenizer_pca",
    "subatomic_tokenizer_pca_8",
    "subatomic_tokenizer_pca_16",
    "subatomic_tokenizer_pca_24",
)
_SUPPORTED_SUBATOMIC_TOKENIZER_PCA_DIMS = {8, 16, 24}


def _supported_type_encoding_message() -> str:
    return ", ".join(_SUPPORTED_TYPE_ENCODING_NAMES)


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
        type_indices = torch.where(
            real_mask, a0.long() - 1, torch.full_like(a0.long(), self.vz)
        )
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
    (3, 1, 0, 5, 0),   #  24  Cr  [Ar] 3d^5 4s^1
    (3, 2, 0, 5, 0),   #  25  Mn
    (3, 2, 0, 6, 0),   #  26  Fe
    (3, 2, 0, 7, 0),   #  27  Co
    (3, 2, 0, 8, 0),   #  28  Ni
    (3, 1, 0, 10, 0),  #  29  Cu  [Ar] 3d^10 4s^1
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
    (4, 1, 0, 4, 0),   #  41  Nb  [Kr] 4d^4 5s^1
    (4, 1, 0, 5, 0),   #  42  Mo  [Kr] 4d^5 5s^1
    (4, 2, 0, 5, 0),   #  43  Tc
    (4, 1, 0, 7, 0),   #  44  Ru  [Kr] 4d^7 5s^1
    (4, 1, 0, 8, 0),   #  45  Rh  [Kr] 4d^8 5s^1
    (4, 0, 0, 10, 0),  #  46  Pd  [Kr] 4d^10
    (4, 1, 0, 10, 0),  #  47  Ag  [Kr] 4d^10 5s^1
    (4, 2, 0, 10, 0),  #  48  Cd
    (4, 2, 1, 10, 0),  #  49  In
    (4, 2, 2, 10, 0),  #  50  Sn
    (4, 2, 3, 10, 0),  #  51  Sb
    (4, 2, 4, 10, 0),  #  52  Te
    (4, 2, 5, 10, 0),  #  53  I
    (4, 2, 6, 10, 0),  #  54  Xe
    (5, 1, 0, 0, 0),   #  55  Cs
    (5, 2, 0, 0, 0),   #  56  Ba
    (5, 2, 0, 1, 0),   #  57  La  [Xe] 5d^1 6s^2
    (5, 2, 0, 1, 1),   #  58  Ce  [Xe] 4f^1 5d^1 6s^2
    (5, 2, 0, 0, 3),   #  59  Pr  [Xe] 4f^3 6s^2
    (5, 2, 0, 0, 4),   #  60  Nd
    (5, 2, 0, 0, 5),   #  61  Pm
    (5, 2, 0, 0, 6),   #  62  Sm
    (5, 2, 0, 0, 7),   #  63  Eu
    (5, 2, 0, 1, 7),   #  64  Gd  [Xe] 4f^7 5d^1 6s^2
    (5, 2, 0, 0, 9),   #  65  Tb  [Xe] 4f^9 6s^2
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
    (5, 1, 0, 9, 14),  #  78  Pt  [Xe] 4f^14 5d^9 6s^1
    (5, 1, 0, 10, 14), #  79  Au  [Xe] 4f^14 5d^10 6s^1
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
    (6, 2, 0, 1, 2),   #  91  Pa  [Rn] 5f^2 6d^1 7s^2
    (6, 2, 0, 1, 3),   #  92  U   [Rn] 5f^3 6d^1 7s^2
    (6, 2, 0, 1, 4),   #  93  Np  [Rn] 5f^4 6d^1 7s^2
    (6, 2, 0, 0, 6),   #  94  Pu
    (6, 2, 0, 0, 7),   #  95  Am
    (6, 2, 0, 1, 7),   #  96  Cm  [Rn] 5f^7 6d^1 7s^2
    (6, 2, 0, 0, 9),   #  97  Bk
    (6, 2, 0, 0, 10),  #  98  Cf
    (6, 2, 0, 0, 11),  #  99  Es
    (6, 2, 0, 0, 12),  # 100  Fm
    (6, 2, 0, 0, 13),  # 101  Md
    (6, 2, 0, 0, 14),  # 102  No
    (6, 2, 1, 0, 14),  # 103  Lr  [Rn] 5f^14 7s^2 7p^1 (IUPAC 2021)
    (6, 2, 0, 2, 14),  # 104  Rf
    (6, 2, 0, 3, 14),  # 105  Db
    (6, 2, 0, 4, 14),  # 106  Sg
    (6, 2, 0, 5, 14),  # 107  Bh
    (6, 2, 0, 6, 14),  # 108  Hs
    (6, 2, 0, 7, 14),  # 109  Mt
    (6, 2, 0, 8, 14),  # 110  Ds  (predicted)
    (6, 1, 0, 10, 14), # 111  Rg  (predicted, [Rn] 5f^14 6d^10 7s^1)
    (6, 2, 0, 10, 14), # 112  Cn
    (6, 2, 1, 10, 14), # 113  Nh
    (6, 2, 2, 10, 14), # 114  Fl
    (6, 2, 3, 10, 14), # 115  Mc
    (6, 2, 4, 10, 14), # 116  Lv
    (6, 2, 5, 10, 14), # 117  Ts
    (6, 2, 6, 10, 14), # 118  Og
)


_SUBATOMIC_TOKENIZER_PERIOD_DIM = 7
_SUBATOMIC_TOKENIZER_GROUP_DIM = 19  # group 0 reserved for f-block / no standard group
_SUBATOMIC_TOKENIZER_BLOCK_DIM = 4
_SUBATOMIC_TOKENIZER_VALENCE_DIM = 4
_SUBATOMIC_TOKENIZER_BLOCK_TO_INDEX = {"s": 0, "p": 1, "d": 2, "f": 3}
_SUBATOMIC_TOKENIZER_GROUP_SLICES = {
    "period": slice(0, _SUBATOMIC_TOKENIZER_PERIOD_DIM),
    "group": slice(
        _SUBATOMIC_TOKENIZER_PERIOD_DIM,
        _SUBATOMIC_TOKENIZER_PERIOD_DIM + _SUBATOMIC_TOKENIZER_GROUP_DIM,
    ),
    "block": slice(
        _SUBATOMIC_TOKENIZER_PERIOD_DIM + _SUBATOMIC_TOKENIZER_GROUP_DIM,
        _SUBATOMIC_TOKENIZER_PERIOD_DIM
        + _SUBATOMIC_TOKENIZER_GROUP_DIM
        + _SUBATOMIC_TOKENIZER_BLOCK_DIM,
    ),
    "valence": slice(
        _SUBATOMIC_TOKENIZER_PERIOD_DIM
        + _SUBATOMIC_TOKENIZER_GROUP_DIM
        + _SUBATOMIC_TOKENIZER_BLOCK_DIM,
        _SUBATOMIC_TOKENIZER_PERIOD_DIM
        + _SUBATOMIC_TOKENIZER_GROUP_DIM
        + _SUBATOMIC_TOKENIZER_BLOCK_DIM
        + _SUBATOMIC_TOKENIZER_VALENCE_DIM,
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


def _build_subatomic_tokenizer_descriptor_table(
    vz: int,
) -> tuple[int, torch.Tensor, dict[str, slice]]:
    max_z = min(int(vz), len(_GROUND_STATE_EC))
    if max_z <= 0:
        raise ValueError(f"No subatomic tokenizer descriptors available for vz={vz}.")

    period_rows: list[torch.Tensor] = []
    group_rows: list[torch.Tensor] = []
    block_rows: list[torch.Tensor] = []
    valence_rows: list[torch.Tensor] = []
    for z in range(1, max_z + 1):
        el = Element.from_Z(z)
        block = str(el.block).lower()
        if block not in _SUBATOMIC_TOKENIZER_BLOCK_TO_INDEX:
            raise ValueError(f"Unsupported block '{el.block}' for Z={z}.")
        period = int(el.row)
        if period < 1 or period > _SUBATOMIC_TOKENIZER_PERIOD_DIM:
            raise ValueError(f"Unsupported period {period} for Z={z}.")
        group = 0 if block == "f" else int(el.group or 0)
        if group < 0 or group >= _SUBATOMIC_TOKENIZER_GROUP_DIM:
            raise ValueError(f"Unsupported group {group} for Z={z}.")

        _, s, p, d, f = _GROUND_STATE_EC[z - 1]
        period_rows.append(
            F.one_hot(
                torch.tensor(period - 1, dtype=torch.long),
                num_classes=_SUBATOMIC_TOKENIZER_PERIOD_DIM,
            ).to(dtype=torch.float32)
        )
        group_rows.append(
            F.one_hot(
                torch.tensor(group, dtype=torch.long),
                num_classes=_SUBATOMIC_TOKENIZER_GROUP_DIM,
            ).to(dtype=torch.float32)
        )
        block_rows.append(
            F.one_hot(
                torch.tensor(_SUBATOMIC_TOKENIZER_BLOCK_TO_INDEX[block], dtype=torch.long),
                num_classes=_SUBATOMIC_TOKENIZER_BLOCK_DIM,
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
        key: slice(value.start, value.stop)
        for key, value in _SUBATOMIC_TOKENIZER_GROUP_SLICES.items()
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


class SubatomicTokenizerRawEncoding:
    """Normalized subatomic-tokenizer descriptor without PCA.

    Descriptor groups:
      - true period one-hot (7)
      - true group one-hot (19 with index 0 reserved for f-block)
      - true block one-hot (s/p/d/f)
      - valence orbital occupancies (s/2, p/6, d/10, f/14)

    The descriptor is standardized, group-weighted, and finally L2-normalized so
    training geometry and cosine-snap decode use the same notion of proximity.
    """

    def __init__(self, vz: int) -> None:
        self.vz = int(vz)
        self.name = "subatomic_tokenizer_raw"
        (
            self.max_z,
            descriptor_table,
            self.group_slices,
        ) = _build_subatomic_tokenizer_descriptor_table(vz=vz)
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


class SubatomicTokenizerPCAEncoding:
    """PCA-compressed subatomic-tokenizer descriptor."""

    def __init__(self, vz: int, pca_dim: int = 16) -> None:
        self.vz = int(vz)
        self.requested_pca_dim = int(pca_dim)
        if self.requested_pca_dim <= 0:
            raise ValueError(f"pca_dim must be > 0, got {self.requested_pca_dim}.")
        self.name = f"subatomic_tokenizer_pca_{self.requested_pca_dim}"

        (
            self.max_z,
            descriptor_table,
            self.group_slices,
        ) = _build_subatomic_tokenizer_descriptor_table(vz=vz)
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
                "subatomic_tokenizer_pca requires at least one PCA component; "
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
    | SubatomicTokenizerRawEncoding
    | SubatomicTokenizerPCAEncoding
)


def _legacy_type_encoding_error(mode: str) -> ValueError:
    mode_norm = mode.strip().lower()
    if mode_norm == "chem_raw_v2":
        return ValueError(
            "Legacy type encoding 'chem_raw_v2' is no longer supported. "
            "Rename it to 'subatomic_tokenizer_raw'."
        )
    if mode_norm == "chem_pca_v2":
        return ValueError(
            "Legacy type encoding 'chem_pca_v2' is no longer supported. "
            "Rename it to 'subatomic_tokenizer_pca_24'."
        )
    if mode_norm.startswith("chem_pca_v2_"):
        suffix = mode_norm.rsplit("_", 1)[-1]
        if suffix in {"8", "16", "24"}:
            return ValueError(
                f"Legacy type encoding '{mode}' is no longer supported. "
                f"Rename it to 'subatomic_tokenizer_pca_{suffix}'."
            )
        return ValueError(
            f"Legacy type encoding '{mode}' is no longer supported. "
            "Supported PCA modes are: subatomic_tokenizer_pca, "
            "subatomic_tokenizer_pca_8, subatomic_tokenizer_pca_16, "
            "subatomic_tokenizer_pca_24."
        )
    raise AssertionError(f"Unhandled legacy type encoding '{mode}'.")


def _parse_supported_subatomic_tokenizer_pca_dim(mode: str) -> int:
    suffix = mode.rsplit("_", 1)[-1]
    try:
        pca_dim = int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Invalid type encoding '{mode}'. Expected one of: {_supported_type_encoding_message()}."
        ) from exc
    if pca_dim not in _SUPPORTED_SUBATOMIC_TOKENIZER_PCA_DIMS:
        raise ValueError(
            f"Unsupported type encoding '{mode}'. Expected one of: {_supported_type_encoding_message()}."
        )
    return pca_dim


def build_type_encoding(mode: str, vz: int) -> TypeEncoding:
    mode_norm = mode.strip().lower()
    if mode_norm == "atomic_number":
        return AtomicNumberEncoding(vz=vz)
    if mode_norm == "subatomic_tokenizer_raw":
        return SubatomicTokenizerRawEncoding(vz=vz)
    if mode_norm == "subatomic_tokenizer_pca":
        return SubatomicTokenizerPCAEncoding(vz=vz, pca_dim=24)
    if mode_norm.startswith("subatomic_tokenizer_pca_"):
        return SubatomicTokenizerPCAEncoding(
            vz=vz, pca_dim=_parse_supported_subatomic_tokenizer_pca_dim(mode_norm)
        )
    if mode_norm == "chem_raw_v2" or mode_norm.startswith("chem_pca_v2"):
        raise _legacy_type_encoding_error(mode)
    raise ValueError(
        f"Unknown type encoding '{mode}'. Expected one of: {_supported_type_encoding_message()}."
    )
