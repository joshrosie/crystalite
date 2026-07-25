"""Tests for space-group conditioning: LoRA-on-AdaLN, PropEncoder, and CFG.

Covers the invariants the design relies on:
  * LoRA is a no-op at init (zero-init B) => wrapping a trained Linear is exact.
  * A frozen base checkpoint loads onto the LoRA-wrapped model (key remap) with
    only conditioning params fresh.
  * Warm-start identity: base weights + zero-init conditioning => the conditional
    model reproduces the unconditional base exactly.
  * classifier-free guidance formula, fast paths, and torus-aware frac handling.
  * the collate function carries the conditioning property into the batch.
"""

import torch

from src.crystalite import CrystaliteModel
from src.crystalite.edm_utils import denoise_edm
from src.crystalite.sampler import denoise_cfg, edm_sampler, wrap_frac
from src.models.lora import (
    LoRALinear,
    wrap_adaln_with_lora,
    remap_base_state_dict_for_lora,
)
from src.models.embeddings import PropEncoder
from src.data.mp20_tokens import collate_mp20_tokens


_CFG = dict(d_model=32, n_heads=4, n_layers=3, vz=100, coord_embed_mode="fourier")


def _build_pair():
    torch.manual_seed(0)
    base = CrystaliteModel(**_CFG).eval()
    cond = CrystaliteModel(
        **_CFG, cond_kind="discrete", cond_vocab_size=231, lora_rank=8, lora_alpha=16.0
    ).eval()
    sd = remap_base_state_dict_for_lora(base.state_dict(), cond)
    missing, unexpected = cond.load_state_dict(sd, strict=False)
    return base, cond, missing, unexpected


def _random_inputs(model, bsz=4, n=6):
    tf = torch.randn(bsz, n, model.type_dim)
    fc = torch.randn(bsz, n, 3)
    lat = torch.randn(bsz, 6)
    pad = torch.zeros(bsz, n, dtype=torch.bool)
    pad[:, n - 2 :] = True
    sigma = torch.rand(bsz) + 0.2
    return tf, fc, lat, pad, sigma


def _denoise_kwargs(tf, fc, lat, pad, sigma, model):
    return dict(
        model=model,
        type_noisy=tf,
        frac_noisy=fc,
        lat_noisy=lat,
        pad_mask=pad,
        sigma=sigma,
        sigma_data_type=1.0,
        sigma_data_coord=0.25,
        sigma_data_lat=1.0,
        sigma_min=0.002,
        sigma_max=80.0,
    )


# --- LoRA -----------------------------------------------------------------


def test_lora_linear_is_noop_at_init():
    base = torch.nn.Linear(16, 24)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)
    # base is frozen; only lora params train
    assert not any(p.requires_grad for p in lora.base.parameters())
    assert lora.lora_A.weight.requires_grad and lora.lora_B.weight.requires_grad


def test_lora_linear_changes_output_when_B_trained():
    base = torch.nn.Linear(16, 24)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.weight.normal_(0, 0.5)
    x = torch.randn(5, 16)
    assert not torch.allclose(lora(x), base(x), atol=1e-4)


def test_wrap_adaln_installs_one_adapter_per_block():
    model = CrystaliteModel(
        **_CFG, cond_kind="discrete", cond_vocab_size=231, lora_rank=4, lora_alpha=8.0
    )
    n = sum(isinstance(b.adaLN_mod[-1], LoRALinear) for b in model.trunk.blocks)
    assert n == len(model.trunk.blocks) == _CFG["n_layers"]


# --- PropEncoder ----------------------------------------------------------


def test_prop_encoder_zero_init():
    enc = PropEncoder(d_model=32, kind="discrete", vocab_size=231)
    out = enc(torch.tensor([0, 225, 1, 139]))
    assert out.shape == (4, 32)
    assert torch.count_nonzero(out) == 0  # zero-init => no signal at construction


# --- base load + warm start ----------------------------------------------


def test_base_load_only_conditioning_params_missing():
    _, _, missing, unexpected = _build_pair()
    assert unexpected == []
    bad = [k for k in missing if "lora_" not in k and "prop_encoder" not in k]
    assert bad == []


def test_warm_start_identity():
    base, cond, _, _ = _build_pair()
    tf, fc, lat, pad, sigma = _random_inputs(base)
    with torch.no_grad():
        o_base = base(tf, fc, lat, pad, sigma, lattice_bias_feats=lat)
        o_null = cond(tf, fc, lat, pad, sigma, lattice_bias_feats=lat, prop=None)
        o_sg = cond(
            tf, fc, lat, pad, sigma, lattice_bias_feats=lat,
            prop=torch.tensor([225, 139, 62, 1]),
        )
    for k in o_base:
        assert torch.allclose(o_base[k], o_null[k], atol=1e-6)
        # zero-init embedding => even a non-null SG is a no-op until trained
        assert torch.allclose(o_base[k], o_sg[k], atol=1e-6)


def test_freeze_selects_only_conditioning_params():
    _, cond, _, _ = _build_pair()
    for name, p in cond.named_parameters():
        p.requires_grad_(("lora_" in name) or ("prop_encoder" in name))
    trainable = {n for n, p in cond.named_parameters() if p.requires_grad}
    assert trainable  # non-empty
    assert all(("lora_" in n) or ("prop_encoder" in n) for n in trainable)
    total = sum(p.numel() for p in cond.parameters())
    trained = sum(p.numel() for p in cond.parameters() if p.requires_grad)
    assert 0 < trained < total


# --- CFG ------------------------------------------------------------------


def _trained_cond_model():
    torch.manual_seed(1)
    m = CrystaliteModel(
        **_CFG, cond_kind="discrete", cond_vocab_size=231, lora_rank=4, lora_alpha=8.0
    ).eval()
    with torch.no_grad():
        m.prop_encoder.embed.weight.normal_(0, 0.5)
        for b in m.trunk.blocks:
            b.adaLN_mod[-1].lora_B.weight.normal_(0, 0.1)
    return m


def test_cfg_fast_paths():
    m = _trained_cond_model()
    tf, fc, lat, pad, sigma = _random_inputs(m)
    kw = _denoise_kwargs(tf, fc, lat, pad, sigma, m)
    target = torch.full((tf.shape[0],), 225, dtype=torch.long)
    null = torch.zeros(tf.shape[0], dtype=torch.long)

    d_cond = denoise_edm(prop=target, **kw)
    d_null = denoise_edm(prop=null, **kw)
    d_none = denoise_edm(prop=None, **kw)

    for k in ("type", "frac", "lat"):
        assert torch.allclose(
            denoise_cfg(guidance_scale=1.0, prop_target=target, **kw)[k], d_cond[k], atol=1e-6
        )
        assert torch.allclose(
            denoise_cfg(guidance_scale=0.0, prop_target=target, **kw)[k], d_null[k], atol=1e-6
        )
        assert torch.allclose(
            denoise_cfg(guidance_scale=3.0, prop_target=None, **kw)[k], d_none[k], atol=1e-6
        )


def test_cfg_formula_with_torus():
    m = _trained_cond_model()
    tf, fc, lat, pad, sigma = _random_inputs(m)
    kw = _denoise_kwargs(tf, fc, lat, pad, sigma, m)
    target = torch.full((tf.shape[0],), 225, dtype=torch.long)
    null = torch.zeros(tf.shape[0], dtype=torch.long)
    d_cond = denoise_edm(prop=target, **kw)
    d_null = denoise_edm(prop=null, **kw)
    # conditioning must actually change the output for this to be meaningful
    assert (d_cond["type"] - d_null["type"]).abs().max() > 1e-3

    w = 2.5
    out = denoise_cfg(guidance_scale=w, prop_target=target, **kw)
    assert torch.allclose(out["type"], d_null["type"] + w * (d_cond["type"] - d_null["type"]), atol=1e-6)
    assert torch.allclose(out["lat"], d_null["lat"] + w * (d_cond["lat"] - d_null["lat"]), atol=1e-6)
    frac_delta = wrap_frac(d_cond["frac"] - d_null["frac"])
    assert torch.allclose(out["frac"], d_null["frac"] + w * frac_delta, atol=1e-6)


def test_unconditional_model_ignores_guidance():
    torch.manual_seed(2)
    m = CrystaliteModel(**_CFG).eval()
    _, _, _, pad, _ = _random_inputs(m)
    common = dict(
        model=m, pad_mask=pad, type_dim=m.type_dim, num_steps=4,
        sigma_min=0.002, sigma_max=80.0, rho=7.0, S_churn=0.0, S_min=0.0,
        S_max=999.0, S_noise=1.0, sigma_data_type=1.0, sigma_data_coord=0.25,
        sigma_data_lat=1.0,
    )
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = edm_sampler(**common, generator=g1, target_spacegroup=225, guidance_scale=5.0)
    b = edm_sampler(**common, generator=g2, target_spacegroup=0, guidance_scale=0.0)
    for k in ("type", "frac", "lat"):
        assert torch.allclose(a[k], b[k], atol=1e-6)


# --- collate --------------------------------------------------------------


def test_collate_carries_property_key():
    def item(sg):
        return {
            "mp_id": "x",
            "A0": torch.zeros(20, dtype=torch.long),
            "F1": torch.zeros(20, 3),
            "Y1": torch.zeros(6),
            "pad_mask": torch.ones(20, dtype=torch.bool),
            "num_atoms": 3,
            "spacegroup.number": sg,
        }

    batch = collate_mp20_tokens([item(225), item(139), item(62)])
    assert "spacegroup.number" in batch
    sg = batch["spacegroup.number"]
    assert torch.is_tensor(sg) and sg.dtype == torch.long
    assert sg.tolist() == [225, 139, 62]


def test_collate_without_property_unchanged():
    def item():
        return {
            "mp_id": "x",
            "A0": torch.zeros(20, dtype=torch.long),
            "F1": torch.zeros(20, 3),
            "Y1": torch.zeros(6),
            "pad_mask": torch.ones(20, dtype=torch.bool),
            "num_atoms": 3,
        }

    batch = collate_mp20_tokens([item(), item()])
    assert set(batch.keys()) == {"mp_id", "A0", "F1", "Y1", "pad_mask", "num_atoms"}
