"""Explicit TorchScript `.nequip.pth` models serve both NequIP relax modes.

The released DNG checkpoint (W&B ai_ecg/DNG_MP20/kny9rodv) was evaluated during training with a TorchScript
NequIP-OAM-L model in batch mode on RTX 6000 Ada GPUs, where AOTInductor `.nequip.pt2` builds crash. These tests
load no model: they check artifact resolution and that the relaxer factories hand the `.pth` to the calculators.
"""
import pytest

import src.utils.sample_stats as sample_stats
from src.utils.sample_stats import (
    make_nequip_batch_relaxer,
    make_nequip_relaxer,
    resolve_nequip_compiled_model,
)


def _model(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"placeholder")
    return path


@pytest.fixture
def no_aot_inspection(monkeypatch):
    calls = []

    def inspect(path):
        calls.append(str(path))
        raise AssertionError("TorchScript models must not be inspected as AOTInductor packages")

    monkeypatch.setattr(sample_stats, "_inspect_nequip_compiled_model", inspect)
    return calls


@pytest.mark.parametrize("target", [None, "ase", "batch"])
def test_explicit_torchscript_model_resolves_for_every_target(tmp_path, no_aot_inspection, target):
    model = _model(tmp_path, "nequip_oaml.nequip.pth")
    assert resolve_nequip_compiled_model(model, expected_target=target) == str(model.resolve())
    assert resolve_nequip_compiled_model(str(model), expected_target=target) == str(model.resolve())
    assert no_aot_inspection == []


def test_explicit_aot_model_still_checks_its_target(tmp_path, monkeypatch):
    model = _model(tmp_path, "NequIP-OAM-L-batch.nequip.pt2")
    monkeypatch.setattr(sample_stats, "_inspect_nequip_compiled_model",
                        lambda path: {"path": str(path), "target": "batch", "input_keys": ()})
    assert resolve_nequip_compiled_model(model, expected_target="batch") == str(model.resolve())
    with pytest.raises(ValueError, match="target mismatch"):
        resolve_nequip_compiled_model(model, expected_target="ase")


@pytest.mark.parametrize("name", ["model.pt", "model.pth", "model.nequip.zip"])
def test_other_explicit_files_are_rejected_naming_both_formats(tmp_path, name):
    model = _model(tmp_path, name)
    with pytest.raises(ValueError) as excinfo:
        resolve_nequip_compiled_model(model, expected_target="batch")
    assert ".nequip.pt2" in str(excinfo.value) and ".nequip.pth" in str(excinfo.value)


def test_directories_and_globs_still_resolve_aot_artifacts_only(tmp_path, monkeypatch):
    _model(tmp_path, "a.nequip.pth")
    aot = _model(tmp_path, "b.nequip.pt2")
    monkeypatch.setattr(sample_stats, "_inspect_nequip_compiled_model",
                        lambda path: {"path": str(path), "target": "batch", "input_keys": ()})
    assert resolve_nequip_compiled_model(tmp_path, expected_target="batch") == str(aot.resolve())
    assert resolve_nequip_compiled_model(str(tmp_path / "*"), expected_target="batch") == str(aot.resolve())


def test_batch_and_sequential_relaxers_load_the_torchscript_model(tmp_path, monkeypatch, no_aot_inspection):
    model = _model(tmp_path, "nequip_oaml.nequip.pth")
    loaded = []

    def calculator(*, compiled_model, stability_device):
        loaded.append(compiled_model)
        return object(), stability_device

    monkeypatch.setattr(sample_stats, "make_nequip_batch_calculator", calculator)
    monkeypatch.setattr(sample_stats, "make_nequip_ase_calculator", calculator)
    _, batch_relaxer, _, batch_path = make_nequip_batch_relaxer(
        compile_path=model, stability_device="cpu", cell_filter="frechet")
    _, ase_relaxer, _, ase_path = make_nequip_relaxer(
        compile_path=str(model), stability_device="cpu", cell_filter="frechet", fmax=0.005)
    assert batch_path == ase_path == str(model.resolve())
    assert loaded == [str(model.resolve()), str(model.resolve())]
    assert batch_relaxer.cell_filter == "frechet" and ase_relaxer.fmax == 0.005
    assert no_aot_inspection == []
