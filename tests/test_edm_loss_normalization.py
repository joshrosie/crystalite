import torch

from src.crystalite.edm_utils import compute_edm_loss


def _zero_clean_and_noisy(
    batch: int, nmax: int, type_dims: int
) -> tuple[dict, dict, torch.Tensor]:
    clean = {
        "type": torch.zeros((batch, nmax, type_dims), dtype=torch.float32),
        "frac_c": torch.zeros((batch, nmax, 3), dtype=torch.float32),
        "lat": torch.zeros((batch, 6), dtype=torch.float32),
    }
    denoised = {
        "type": torch.zeros_like(clean["type"]),
        "frac": torch.zeros_like(clean["frac_c"]),
        "lat": torch.zeros_like(clean["lat"]),
    }
    frac_noisy = torch.zeros_like(clean["frac_c"])
    return denoised, clean, frac_noisy


def test_type_loss_is_scalar_mean_over_channels():
    denoised, clean, frac_noisy = _zero_clean_and_noisy(batch=1, nmax=3, type_dims=5)
    pad_mask = torch.tensor([[False, False, True]])
    sigma = torch.tensor([1.0], dtype=torch.float32)

    denoised["type"][0, :2, :] = 1.0

    losses = compute_edm_loss(
        denoised=denoised,
        clean=clean,
        frac_noisy=frac_noisy,
        sigma=sigma,
        pad_mask=pad_mask,
        sigma_data_type=1.0,
        sigma_data_coord=1.0,
        sigma_data_lat=1.0,
        loss_weights={"A": 1.0, "F": 1.0, "Y": 1.0},
    )

    torch.testing.assert_close(losses["loss_a"], torch.tensor(2.0))
    torch.testing.assert_close(losses["loss_f"], torch.tensor(0.0))
    torch.testing.assert_close(losses["loss_y"], torch.tensor(0.0))


def test_frac_coord_loss_is_scalar_mean_over_xyz():
    denoised, clean, frac_noisy = _zero_clean_and_noisy(batch=1, nmax=3, type_dims=4)
    pad_mask = torch.tensor([[False, True, False]])
    sigma = torch.tensor([1.0], dtype=torch.float32)

    denoised["frac"][0, 0, :] = 0.25
    denoised["frac"][0, 2, :] = 0.25

    losses = compute_edm_loss(
        denoised=denoised,
        clean=clean,
        frac_noisy=frac_noisy,
        sigma=sigma,
        pad_mask=pad_mask,
        sigma_data_type=1.0,
        sigma_data_coord=1.0,
        sigma_data_lat=1.0,
        loss_weights={"A": 1.0, "F": 1.0, "Y": 1.0},
        coord_loss_mode="frac_mse",
    )

    torch.testing.assert_close(losses["loss_f"], torch.tensor(0.125))
    torch.testing.assert_close(losses["loss_a"], torch.tensor(0.0))
    torch.testing.assert_close(losses["loss_y"], torch.tensor(0.0))
