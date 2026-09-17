"""Portable model-owned encodings. PCA is fitted only for fresh training.

Keep this format compatible with Crysfinity. Checkpoint consumers must use
``resolve_type_encoding`` rather than rebuilding an encoding from its name.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .type_encoding import (
    AtomicNumberEncoding,
    SubatomicTokenizerPCAEncoding,
    SubatomicTokenizerRawEncoding,
    TypeEncoding,
    build_type_encoding,
)

FORMAT_VERSION = 1
ASSET_DIR = Path(__file__).with_name("encoding_assets")
_COMMON = {"vz", "name", "type_dim"}
_DESCRIPTOR = {
    "max_z", "group_slices", "descriptor_dim", "descriptor_mean", "descriptor_std",
    "weighted_descriptor_table", "prototype_norms_before_normalization",
    "z_to_enc", "snap_table",
}
_PCA = {
    "requested_pca_dim", "pca_components", "explained_variance_ratio",
    "cumulative_explained_variance",
}


def canonical_encoding_name(name: str) -> str:
    name = str(name).strip().lower()
    if name == "chem_raw_v2":
        name = "subatomic_tokenizer_raw"
    if name == "chem_pca_v2":
        name = "subatomic_tokenizer_pca_24"
    if name.startswith("chem_pca_v2_"):
        name = name.replace("chem_pca_v2_", "subatomic_tokenizer_pca_", 1)
    if name == "subatomic_tokenizer_pca":
        name = "subatomic_tokenizer_pca_24"
    if name not in {"atomic_number", "subatomic_tokenizer_raw", *(
        f"subatomic_tokenizer_pca_{d}" for d in (8, 16, 24)
    )}:
        raise ValueError(f"Unsupported type encoding {name!r}.")
    return name


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _hashable(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": tensor_sha256(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Encoding/fingerprint mappings require string keys.")
        return {k: _hashable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported encoding/fingerprint value: {type(value).__name__}")


def _sha256(value: Any) -> str:
    payload = json.dumps(_hashable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_weights_sha256(ckpt: Mapping[str, Any]) -> str:
    """Fingerprint model + EMA tensors, independent of filenames and metadata."""
    if not isinstance(ckpt.get("model_state_dict"), Mapping):
        raise ValueError("Checkpoint is missing model_state_dict for encoding verification.")
    return _sha256({k: ckpt.get(k) for k in ("model_state_dict", "ema_state_dict")})


def _state_digest(state: Mapping[str, Any]) -> str:
    return _sha256({k: state[k] for k in ("format_version", "encoding_state", "element_order")})


def _fields(name: str) -> set[str]:
    if name == "atomic_number":
        return _COMMON
    return _COMMON | _DESCRIPTOR | (_PCA if "_pca_" in name else set())


def export_type_encoding(encoding: TypeEncoding, *, provenance: dict | None = None) -> dict:
    """Freeze the actual state in use; never refit or renormalize it."""
    name = canonical_encoding_name(encoding.name)
    values = {}
    for key in _fields(name):
        value = getattr(encoding, key)
        if torch.is_tensor(value):
            value = value.detach().cpu().clone()
        elif key == "group_slices":
            value = {k: [v.start, v.stop, v.step] for k, v in value.items()}
        values[key] = value
    values["name"] = name
    state = {
        "format_version": FORMAT_VERSION,
        "encoding_state": values,
        "element_order": list(range(1, int(values.get("max_z", values["vz"])) + 1)),
        "provenance": dict(provenance if provenance is not None else getattr(
            encoding, "_encoding_provenance", {"source": "training"}
        )),
    }
    state["state_sha256"] = _state_digest(state)
    return state


def restore_type_encoding(state: Mapping[str, Any]) -> TypeEncoding:
    """Validate and restore without calling descriptor construction or PCA."""
    if not isinstance(state, Mapping) or state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported or missing type_encoding_state format_version.")
    if not isinstance(state.get("encoding_state"), Mapping) or "element_order" not in state:
        raise ValueError("Incomplete type_encoding_state.")
    if state.get("state_sha256") != _state_digest(state):
        raise ValueError("type_encoding_state hash mismatch; state is corrupt or modified.")
    values = dict(state["encoding_state"])
    name = canonical_encoding_name(values.get("name", ""))
    if set(values) != _fields(name):
        raise ValueError(f"Incomplete or unexpected fields in {name} state.")
    for key in ("vz", "type_dim"):
        if type(values[key]) is not int or values[key] <= 0:
            raise ValueError(f"Invalid encoding {key}.")
    vz, dim = values["vz"], values["type_dim"]
    if name == "atomic_number":
        if dim != vz + 1 or state["element_order"] != list(range(1, vz + 1)):
            raise ValueError("Invalid atomic_number dimensions or element ordering.")
        encoding = AtomicNumberEncoding(vz)
    else:
        max_z = min(vz, 118)
        if values["max_z"] != max_z or values["descriptor_dim"] != 34:
            raise ValueError("Invalid descriptor dimensions or element range.")
        if state["element_order"] != list(range(1, max_z + 1)):
            raise ValueError("Invalid encoding element ordering.")
        expected_slices = {"period": [0, 7, None], "group": [7, 26, None],
                           "block": [26, 30, None], "valence": [30, 34, None]}
        if values["group_slices"] != expected_slices:
            raise ValueError("Invalid descriptor group slices.")
        shapes = {
            "descriptor_mean": (34,), "descriptor_std": (34,),
            "weighted_descriptor_table": (max_z, 34),
            "prototype_norms_before_normalization": (max_z,),
            "z_to_enc": (max_z + 1, dim), "snap_table": (max_z, dim),
        }
        if "_pca_" in name:
            requested = int(name.rsplit("_", 1)[1])
            if values["requested_pca_dim"] != requested or dim != min(requested, max_z, 34):
                raise ValueError("Invalid PCA dimensions.")
            shapes.update(pca_components=(34, dim), explained_variance_ratio=(34,),
                          cumulative_explained_variance=(34,))
            cls = SubatomicTokenizerPCAEncoding
        else:
            if dim != 34:
                raise ValueError("Invalid raw descriptor dimension.")
            cls = SubatomicTokenizerRawEncoding
        for key, shape in shapes.items():
            tensor = values[key]
            if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid encoding tensor {key}; expected finite float32 {shape}.")
            values[key] = tensor.detach().cpu().clone()
        if (values["descriptor_std"] <= 0).any():
            raise ValueError("Descriptor standard deviations must be positive.")
        if (torch.count_nonzero(values["z_to_enc"][0])
                or not torch.equal(values["z_to_enc"][1:], values["snap_table"])):
            raise ValueError("Encoding and decoding lookup tables disagree.")
        values["group_slices"] = {k: slice(*v) for k, v in values["group_slices"].items()}
        encoding = cls.__new__(cls)  # Deliberately bypass __init__: it fits PCA.
        encoding.__dict__.update(values)
    encoding.name = name
    encoding._encoding_provenance = dict(state.get("provenance", {}))
    encoding._encoding_source = "checkpoint"
    return encoding


def type_encoding_metadata(encoding: TypeEncoding) -> dict:
    state = export_type_encoding(encoding)
    result = {
        "name": encoding.name, "source": getattr(encoding, "_encoding_source", "training"),
        "format_version": FORMAT_VERSION, "state_sha256": state["state_sha256"],
    }
    if hasattr(encoding, "snap_table"):
        result["snap_table_sha256"] = tensor_sha256(encoding.snap_table)
    return result


def _check_metadata(ckpt: Mapping[str, Any], encoding: TypeEncoding, vz: int) -> None:
    if encoding.vz != vz:
        raise ValueError(f"Checkpoint encoding vocabulary {encoding.vz} != runtime vocabulary {vz}.")
    for config in (ckpt, ckpt.get("model_args", {})):
        if config.get("type_encoding") is not None:
            if canonical_encoding_name(config["type_encoding"]) != encoding.name:
                raise ValueError("Checkpoint has conflicting type_encoding metadata and saved state.")
        if config.get("type_dim") is not None and int(config["type_dim"]) != encoding.type_dim:
            raise ValueError("Checkpoint has conflicting type_dim metadata and saved state.")


def resolve_type_encoding(
    ckpt: Mapping[str, Any], *, vz: int, verified_state: Mapping[str, Any] | None = None,
) -> TypeEncoding:
    """Embedded state, explicitly verified recovery, known M0, or legacy non-PCA.

    An explicit recovery artifact must bind its provenance to this checkpoint's
    model/EMA fingerprint. That binding records the recovery author's verification;
    a checksum alone cannot establish the historical scientific provenance.
    """
    embedded = ckpt.get("type_encoding_state")
    if embedded is not None:
        encoding = restore_type_encoding(embedded)
        if verified_state is not None:
            supplied = restore_type_encoding(verified_state)
            if export_type_encoding(supplied)["state_sha256"] != embedded["state_sha256"]:
                raise ValueError("Explicit encoding state conflicts with embedded checkpoint state.")
    elif verified_state is not None:
        encoding = restore_type_encoding(verified_state)
        expected = verified_state.get("provenance", {}).get("checkpoint_weights_sha256")
        if expected != checkpoint_weights_sha256(ckpt):
            raise ValueError("Recovery state needs a verified checkpoint_weights_sha256 matching this checkpoint.")
        encoding._encoding_source = "explicit_verified_state"
    else:
        args = ckpt.get("model_args", {})
        name = canonical_encoding_name(ckpt.get("type_encoding", args.get("type_encoding", "atomic_number")))
        if "_pca_" in name:
            registry = json.loads((ASSET_DIR / "registry.json").read_text())
            fingerprint = checkpoint_weights_sha256(ckpt)
            entry = registry.get(fingerprint)
            if entry is None:
                raise ValueError(
                    "Legacy PCA checkpoint has no verified type_encoding_state. Refusing to refit PCA. "
                    "Recover the original table and migrate with --type-encoding-state; "
                    "see docs/type_encoding.md."
                )
            state = torch.load(ASSET_DIR / entry["asset"], map_location="cpu", weights_only=True)
            encoding = restore_type_encoding(state)
            if (state["state_sha256"] != entry["state_sha256"] or
                    state.get("provenance", {}).get("checkpoint_weights_sha256") != fingerprint):
                raise ValueError("Bundled encoding asset does not match its verified registry entry.")
            encoding._encoding_source = "bundled_verified_m0"
        else:
            encoding = build_type_encoding(name, vz=vz)
            encoding._encoding_source = "legacy_non_pca"
            encoding._encoding_provenance = {"source": "legacy_non_pca"}
    _check_metadata(ckpt, encoding, vz)
    return encoding


def migrate_type_encoding(ckpt: Mapping[str, Any], *, vz: int, verified_state: Mapping | None = None) -> dict:
    """Attach verified state without changing model/EMA tensors or the source dict."""
    encoding = resolve_type_encoding(ckpt, vz=vz, verified_state=verified_state)
    out = dict(ckpt)
    out["model_args"] = dict(ckpt.get("model_args", {}))
    for config in (out, out["model_args"]):
        config.update(type_encoding=encoding.name, type_dim=encoding.type_dim)
    out["type_encoding_state"] = export_type_encoding(encoding)
    return out


def training_type_encoding(
    name: str, *, vz: int, init_checkpoint: Mapping | None = None, explicit_name: bool = False,
) -> TypeEncoding:
    """Fresh runs fit their own state; fine-tuning inherits the parent model's."""
    if init_checkpoint is None:
        return build_type_encoding(name, vz=vz)
    encoding = resolve_type_encoding(init_checkpoint, vz=vz)
    if explicit_name and canonical_encoding_name(name) != encoding.name:
        raise ValueError("Explicit --type_encoding conflicts with --init_from checkpoint encoding.")
    return encoding


def save_migrated_checkpoint(ckpt: Mapping, *, source: Path, destination: Path, overwrite: bool = False) -> None:
    """Preserve source files and publish only a fully serialized destination."""
    import os
    import tempfile

    if source.resolve() == destination.resolve() or (destination.exists() and source.samefile(destination)):
        raise ValueError("Use a different destination; never overwrite the source checkpoint.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {destination}. Use --overwrite to replace it.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(dict(ckpt), stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)  # Exclusive creation, even if another process raced us.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
