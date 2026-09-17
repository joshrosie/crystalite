from __future__ import annotations

import copy
import json

import pytest
import torch

from src.models import type_encoding as constructors
from src.models import type_encoding_state as state_io
from src.utils.ema import EMA


def checkpoint(encoding):
    return {
        "model_args": {"type_encoding": encoding.name},
        "type_encoding": encoding.name, "type_dim": encoding.type_dim,
        "model_state_dict": {"weight": torch.tensor([1., 2.])},
        "ema_state_dict": {"weight": torch.tensor([3., 4.])},
        "type_encoding_state": state_io.export_type_encoding(encoding),
    }


def forbid_refitting(*args, **kwargs):
    raise AssertionError("Loading an encoding must not refit PCA or rebuild descriptors")


@pytest.mark.parametrize("name", ["atomic_number", "subatomic_tokenizer_raw", *(
    f"subatomic_tokenizer_pca_{d}" for d in (8, 16, 24)
)])
def test_portable_roundtrip_uses_actual_table_without_constructor(name, monkeypatch, tmp_path):
    original = constructors.build_type_encoding(name, vz=94)
    original_state = state_io.export_type_encoding(original)
    path = tmp_path / "encoding.pt"
    torch.save(original_state, path)
    monkeypatch.setattr(constructors, "_build_subatomic_tokenizer_descriptor_table", forbid_refitting)
    monkeypatch.setattr(torch.linalg, "eigh", forbid_refitting)
    restored = state_io.restore_type_encoding(torch.load(path, weights_only=True))
    a0 = torch.tensor([[1, 6, 8, 94, 0]])
    padding = a0 == 0
    encoded = original.encode_from_A0(a0, padding)
    assert torch.equal(restored.encode_from_A0(a0, padding), encoded)
    allowed = torch.ones(94, dtype=torch.bool)
    allowed[5] = False
    assert torch.equal(original.decode_logits_to_A0(encoded, padding, allowed),
                       restored.decode_logits_to_A0(encoded, padding, allowed))
    assert state_io.export_type_encoding(restored)["state_sha256"] == original_state["state_sha256"]


def test_frozen_table_is_exact_across_threads(monkeypatch):
    asset = torch.load(state_io.ASSET_DIR / "m0_pca16_v1.pt", weights_only=True)
    original_threads = torch.get_num_threads()
    monkeypatch.setattr(torch.linalg, "eigh", forbid_refitting)
    try:
        for threads in (1, 4, 8, 16, 32):
            torch.set_num_threads(threads)
            encoding = state_io.restore_type_encoding(asset)
            assert torch.equal(encoding.snap_table, asset["encoding_state"]["snap_table"])
            assert state_io.tensor_sha256(encoding.snap_table) == "b8cc20385c514aa45de6ee0068776fedd7d72d3981bcc9507e5de1e267cfb9e1"
    finally:
        torch.set_num_threads(original_threads)


def test_unknown_legacy_pca_fails_but_legacy_atomic_number_works(monkeypatch):
    ckpt = checkpoint(constructors.build_type_encoding("subatomic_tokenizer_pca_16", vz=94))
    del ckpt["type_encoding_state"]
    monkeypatch.setattr(torch.linalg, "eigh", forbid_refitting)
    with pytest.raises(ValueError, match="Refusing to refit PCA"):
        state_io.resolve_type_encoding(ckpt, vz=94)
    atomic = state_io.resolve_type_encoding({"type_encoding": "atomic_number", "type_dim": 95}, vz=94)
    assert atomic.type_dim == 95


def test_registry_requires_exact_weights_and_metadata(monkeypatch):
    registry = json.loads((state_io.ASSET_DIR / "registry.json").read_text())
    fingerprint = next(iter(registry))
    monkeypatch.setattr(state_io, "checkpoint_weights_sha256", lambda ckpt: fingerprint)
    ckpt = {"type_encoding": "chem_pca_v2_16", "type_dim": 16}
    encoding = state_io.resolve_type_encoding(ckpt, vz=94)
    assert state_io.type_encoding_metadata(encoding)["source"] == "bundled_verified_m0"
    ckpt["type_dim"] = 24
    with pytest.raises(ValueError, match="type_dim"):
        state_io.resolve_type_encoding(ckpt, vz=94)


def test_migration_needs_verified_binding_and_preserves_weights():
    encoder = constructors.build_type_encoding("subatomic_tokenizer_pca_8", vz=94)
    ckpt = checkpoint(encoder)
    supplied = ckpt.pop("type_encoding_state")
    with pytest.raises(ValueError, match="checkpoint_weights_sha256"):
        state_io.migrate_type_encoding(ckpt, vz=94, verified_state=supplied)
    supplied["provenance"]["checkpoint_weights_sha256"] = state_io.checkpoint_weights_sha256(ckpt)
    migrated = state_io.migrate_type_encoding(ckpt, vz=94, verified_state=supplied)
    assert migrated["model_state_dict"] is ckpt["model_state_dict"]
    assert migrated["ema_state_dict"] is ckpt["ema_state_dict"]
    assert "type_encoding_state" not in ckpt
    assert migrated["type_encoding_state"]["state_sha256"] == supplied["state_sha256"]
    altered = copy.deepcopy(ckpt)
    altered["ema_state_dict"]["weight"][0] += 1
    with pytest.raises(ValueError, match="checkpoint_weights_sha256"):
        state_io.migrate_type_encoding(altered, vz=94, verified_state=supplied)


@pytest.mark.parametrize("change", ["hash", "shape", "dtype", "ordering", "version", "nan", "lookup"])
def test_bad_embedded_state_never_falls_back(change, monkeypatch):
    ckpt = checkpoint(constructors.build_type_encoding("subatomic_tokenizer_pca_16", vz=94))
    state = ckpt["type_encoding_state"]
    values = state["encoding_state"]
    if change == "hash":
        state["state_sha256"] = "bad"
    elif change == "shape":
        values["snap_table"] = values["snap_table"][:-1]
    elif change == "dtype":
        values["snap_table"] = values["snap_table"].double()
    elif change == "ordering":
        state["element_order"].reverse()
    elif change == "version":
        state["format_version"] = 999
    elif change == "nan":
        values["pca_components"][0, 0] = float("nan")
    elif change == "lookup":
        values["z_to_enc"][1, 0] += 1
    if change != "hash":
        state["state_sha256"] = state_io._state_digest(state)
    monkeypatch.setattr(torch.linalg, "eigh", forbid_refitting)
    monkeypatch.setattr(state_io, "checkpoint_weights_sha256", forbid_refitting)
    with pytest.raises(ValueError):
        state_io.resolve_type_encoding(ckpt, vz=94)


def test_scratch_owns_its_state_and_finetuning_restores_it(monkeypatch):
    own = constructors.build_type_encoding("subatomic_tokenizer_pca_8", vz=94)
    monkeypatch.setattr(state_io, "build_type_encoding", lambda *a, **kw: own)
    fresh = state_io.training_type_encoding(own.name, vz=94)
    assert fresh is own
    ckpt = checkpoint(fresh)
    monkeypatch.setattr(state_io, "build_type_encoding", forbid_refitting)
    restored = state_io.training_type_encoding("atomic_number", vz=94, init_checkpoint=ckpt)
    assert torch.equal(restored.z_to_enc, own.z_to_enc)
    with pytest.raises(ValueError, match="Explicit --type_encoding"):
        state_io.training_type_encoding("atomic_number", vz=94, init_checkpoint=ckpt, explicit_name=True)


def test_migration_preserves_source_and_existing_destination(tmp_path):
    source, target = tmp_path / "source.pt", tmp_path / "target.pt"
    torch.save({"weight": torch.ones(1)}, source)
    with pytest.raises(ValueError, match="never overwrite"):
        state_io.save_migrated_checkpoint({}, source=source, destination=source, overwrite=True)
    state_io.save_migrated_checkpoint({"new": 1}, source=source, destination=target)
    with pytest.raises(FileExistsError):
        state_io.save_migrated_checkpoint({}, source=source, destination=target)
    assert torch.load(target, weights_only=True) == {"new": 1}
    assert torch.equal(torch.load(source, weights_only=True)["weight"], torch.ones(1))


def test_ema_restore_copies_saved_values_and_rejects_incompatible_state():
    model = torch.nn.Linear(3, 2)
    ema = EMA(model, 0.99)
    saved = {k: torch.full_like(v, 0.25) for k, v in ema.state_dict().items()}
    ema.load_state_dict(saved)
    saved["weight"].zero_()
    assert torch.equal(ema.shadow["weight"], torch.full_like(model.weight, 0.25))
    with pytest.raises(ValueError):
        ema.load_state_dict({})


@pytest.mark.parametrize("entrypoint", ["src.sample_crystalite_ckpt", "src.eval_crystalite_ckpt", "src.eval_csp_ckpt"])
def test_inference_rejects_unknown_pca_before_allocating_model(entrypoint, monkeypatch):
    import importlib
    from types import SimpleNamespace

    if importlib.util.find_spec(entrypoint) is None:
        pytest.skip("This repository does not provide the standalone CSP evaluator.")
    module = importlib.import_module(entrypoint)
    ckpt = checkpoint(constructors.build_type_encoding("subatomic_tokenizer_pca_16", vz=94))
    del ckpt["type_encoding_state"]
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(
        checkpoint="unused", train_output_dir="", checkpoint_preference="auto"))
    monkeypatch.setattr(module, "_resolve_checkpoint_path", lambda **kw: "unused")
    monkeypatch.setattr(module, "_load_checkpoint", lambda path: ckpt)
    monkeypatch.setattr(module, "_build_model_from_ckpt", forbid_refitting)
    with pytest.raises(ValueError, match="Refusing to refit PCA"):
        module.main()
