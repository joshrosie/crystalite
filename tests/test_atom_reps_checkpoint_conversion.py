from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from src.convert_atom_reps_ckpt import (
    build_crystalite_model_from_ckpt,
    convert_atom_reps_checkpoint,
    load_checkpoint,
    normalize_atom_reps_type_encoding_name,
    verify_crystalite_load,
)
from src.crystalite import CrystaliteModel
from src.data.mp20_tokens import VZ
from src.models.type_encoding import build_type_encoding
from src.models.type_encoding_state import export_type_encoding


def _type_dim_for(source_type_encoding: str) -> int:
    target_type_encoding = normalize_atom_reps_type_encoding_name(source_type_encoding)
    return int(build_type_encoding(target_type_encoding, vz=VZ).type_dim)


def _minimal_checkpoint(source_type_encoding: str) -> dict:
    return {
        "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "ema_state_dict": {"weight": torch.tensor([3.0, 4.0])},
        "model_args": {"type_encoding": source_type_encoding},
        "type_encoding": source_type_encoding,
        "type_dim": _type_dim_for(source_type_encoding),
        "step": 123,
        "type_encoding_state": export_type_encoding(build_type_encoding(
            normalize_atom_reps_type_encoding_name(source_type_encoding), vz=VZ)),
    }


@pytest.mark.parametrize(
    ("source_type_encoding", "target_type_encoding"),
    [
        ("chem_raw_v2", "subatomic_tokenizer_raw"),
        ("chem_pca_v2", "subatomic_tokenizer_pca_24"),
        ("chem_pca_v2_8", "subatomic_tokenizer_pca_8"),
        ("chem_pca_v2_16", "subatomic_tokenizer_pca_16"),
        ("chem_pca_v2_24", "subatomic_tokenizer_pca_24"),
    ],
)
def test_legacy_type_encoding_names_convert(source_type_encoding, target_type_encoding):
    ckpt = _minimal_checkpoint(source_type_encoding)

    converted = convert_atom_reps_checkpoint(
        ckpt,
        source_path=Path("/tmp/source.pt"),
        converted_at="2026-04-27T00:00:00Z",
    )

    assert converted["type_encoding"] == target_type_encoding
    assert converted["model_args"]["type_encoding"] == target_type_encoding
    assert converted["type_dim"] == _type_dim_for(source_type_encoding)
    assert converted["step"] == ckpt["step"]
    assert converted["conversion_meta"] == {
        "converter": "src.convert_atom_reps_ckpt",
        "source_format": "atom-reps",
        "target_format": "crystalite",
        "source_type_encoding": source_type_encoding,
        "target_type_encoding": target_type_encoding,
        "type_dim": _type_dim_for(source_type_encoding),
        "converted_at": "2026-04-27T00:00:00Z",
        "source_path": "/tmp/source.pt",
    }


@pytest.mark.parametrize(
    "source_type_encoding",
    [
        "periodic_table_2d",
        "table",
        "electron_config",
        "ec",
        "chem_pca",
        "chemical_pca",
        "chem_pca_v2_32",
    ],
)
def test_unsupported_atom_reps_type_encodings_fail(source_type_encoding):
    ckpt = {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "model_args": {"type_encoding": source_type_encoding},
        "type_encoding": source_type_encoding,
        "type_dim": 1,
    }

    with pytest.raises(ValueError, match="Unsupported atom-reps type encoding"):
        convert_atom_reps_checkpoint(ckpt)


def test_conversion_preserves_tensor_payloads_and_does_not_mutate_source():
    model_tensor = torch.randn(2, 3)
    ema_tensor = torch.randn(2, 3)
    ckpt = {
        "model_state_dict": {"weight": model_tensor},
        "ema_state_dict": {"weight": ema_tensor},
        "model_args": {"type_encoding": "chem_pca_v2_16"},
        "type_encoding": "chem_pca_v2_16",
        "type_dim": 16,
    }

    ckpt["type_encoding_state"] = export_type_encoding(build_type_encoding("subatomic_tokenizer_pca_16", vz=VZ))
    converted = convert_atom_reps_checkpoint(ckpt, converted_at="2026-04-27T00:00:00Z")

    assert converted["model_state_dict"]["weight"] is model_tensor
    assert converted["ema_state_dict"]["weight"] is ema_tensor
    assert torch.equal(converted["model_state_dict"]["weight"], model_tensor)
    assert torch.equal(converted["ema_state_dict"]["weight"], ema_tensor)
    assert ckpt["type_encoding"] == "chem_pca_v2_16"
    assert ckpt["model_args"]["type_encoding"] == "chem_pca_v2_16"


def test_tiny_converted_checkpoint_strict_loads_and_matches_forward_outputs():
    torch.manual_seed(0)
    model_args = {
        "type_encoding": "chem_pca_v2_16",
        "type_dim": 16,
        "attn_type": "mha",
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 1,
        "coord_n_freqs": 2,
        "coord_embed_mode": "fourier",
        "coord_head_mode": "direct",
        "coord_rff_dim": None,
        "coord_rff_sigma": 1.0,
        "lattice_embed_mode": "mlp",
        "lattice_rff_dim": 8,
        "lattice_rff_sigma": 5.0,
        "lattice_repr": "y1",
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "use_distance_bias": False,
        "use_edge_bias": False,
        "edge_bias_n_freqs": 2,
        "edge_bias_hidden_dim": 16,
        "edge_bias_n_rbf": 4,
        "edge_bias_rbf_max": 2.0,
        "pbc_radius": 1,
        "dist_slope_init": -1.0,
        "use_noise_gate": True,
        "gem_per_layer": False,
    }
    source_model = CrystaliteModel(
        d_model=model_args["d_model"],
        n_heads=model_args["n_heads"],
        n_layers=model_args["n_layers"],
        vz=VZ,
        type_dim=model_args["type_dim"],
        n_freqs=model_args["coord_n_freqs"],
        coord_embed_mode=model_args["coord_embed_mode"],
        coord_head_mode=model_args["coord_head_mode"],
        coord_rff_dim=model_args["coord_rff_dim"],
        coord_rff_sigma=model_args["coord_rff_sigma"],
        lattice_embed_mode=model_args["lattice_embed_mode"],
        lattice_rff_dim=model_args["lattice_rff_dim"],
        lattice_rff_sigma=model_args["lattice_rff_sigma"],
        lattice_repr=model_args["lattice_repr"],
        dropout=model_args["dropout"],
        attn_dropout=model_args["attn_dropout"],
        use_distance_bias=model_args["use_distance_bias"],
        use_edge_bias=model_args["use_edge_bias"],
        edge_bias_n_freqs=model_args["edge_bias_n_freqs"],
        edge_bias_hidden_dim=model_args["edge_bias_hidden_dim"],
        edge_bias_n_rbf=model_args["edge_bias_n_rbf"],
        edge_bias_rbf_max=model_args["edge_bias_rbf_max"],
        pbc_radius=model_args["pbc_radius"],
        dist_slope_init=model_args["dist_slope_init"],
        use_noise_gate=model_args["use_noise_gate"],
        gem_per_layer=model_args["gem_per_layer"],
    )
    source_model.eval()

    ckpt = {
        "model_state_dict": source_model.state_dict(),
        "ema_state_dict": source_model.state_dict(),
        "model_args": dict(model_args),
        "type_encoding": "chem_pca_v2_16",
        "type_dim": 16,
        "step": 7,
    }
    ckpt["type_encoding_state"] = export_type_encoding(build_type_encoding("subatomic_tokenizer_pca_16", vz=VZ))
    converted = convert_atom_reps_checkpoint(ckpt, converted_at="2026-04-27T00:00:00Z")
    loaded_model = verify_crystalite_load(converted, device="cpu")

    inputs = {
        "type_feats": torch.randn(2, 3, 16),
        "frac_coords": torch.rand(2, 3, 3),
        "lattice_feats": torch.randn(2, 6),
        "pad_mask": torch.tensor([[False, False, True], [False, True, True]]),
        "t_sigma": torch.zeros(2),
    }
    with torch.no_grad():
        expected = source_model(**inputs)
        actual = loaded_model(**inputs)

    assert converted["type_encoding"] == "subatomic_tokenizer_pca_16"
    for key in expected:
        assert torch.equal(actual[key], expected[key])


def test_conflicting_checkpoint_type_encoding_values_fail():
    ckpt = _minimal_checkpoint("chem_pca_v2_16")
    ckpt["model_args"]["type_encoding"] = "chem_pca_v2_24"

    with pytest.raises(ValueError, match="conflicting type_encoding"):
        convert_atom_reps_checkpoint(ckpt)


def _load_python_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_optional_atom_reps_descriptor_table_matches_crystalite():
    atom_reps_root = os.environ.get("ATOM_REPS_ROOT")
    if not atom_reps_root:
        pytest.skip("Set ATOM_REPS_ROOT to compare descriptor tables against atom-reps.")

    atom_type_encoding = _load_python_module(
        "atom_reps_type_encoding_for_crystalite_test",
        Path(atom_reps_root) / "src" / "data" / "type_encoding.py",
    )
    source = atom_type_encoding.build_type_encoding("chem_pca_v2_16", vz=VZ)
    target = build_type_encoding("subatomic_tokenizer_pca_16", vz=VZ)

    assert torch.equal(source.z_to_enc, target.z_to_enc)
    assert torch.equal(source.snap_table, target.snap_table)


def test_optional_real_atom_reps_checkpoint_converts_loads_and_runs_forward():
    checkpoint_path = os.environ.get("ATOM_REPS_CKPT")
    if not checkpoint_path:
        pytest.skip("Set ATOM_REPS_CKPT to run the real checkpoint integration test.")

    ckpt = load_checkpoint(Path(checkpoint_path))
    converted = convert_atom_reps_checkpoint(
        ckpt,
        source_path=Path(checkpoint_path),
        converted_at="2026-04-27T00:00:00Z",
    )
    model = build_crystalite_model_from_ckpt(converted, device="cpu")

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(
            type_feats=torch.randn(1, 3, int(converted["type_dim"])),
            frac_coords=torch.rand(1, 3, 3),
            lattice_feats=torch.randn(1, 6),
            pad_mask=torch.tensor([[False, False, True]]),
            t_sigma=torch.zeros(1),
        )

    assert converted["type_encoding"] == "subatomic_tokenizer_pca_16"
    assert set(out) == {"type_logits", "coord_vel", "lattice_vel"}
