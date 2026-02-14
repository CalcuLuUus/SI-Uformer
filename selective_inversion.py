import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def rfft2c(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    return torch.fft.rfft2(x, norm=norm)


def irfft2c(X: torch.Tensor, s: tuple, norm: str = "ortho") -> torch.Tensor:
    return torch.fft.irfft2(X, s=s, norm=norm)


def pad_to_square(x: torch.Tensor, factor: int = 16):
    B, C, H, W = x.shape
    size = int(math.ceil(max(H, W) / float(factor)) * factor)
    if H == size and W == size:
        pad_info = (0, 0, H, W)
        pad_mask = x.new_ones(B, 1, H, W)
        return x, pad_mask, pad_info

    out = x.new_zeros(B, C, size, size)
    pad_mask = x.new_zeros(B, 1, size, size)
    top = (size - H) // 2
    left = (size - W) // 2
    out[:, :, top:top + H, left:left + W] = x
    pad_mask[:, :, top:top + H, left:left + W] = 1.0
    pad_info = (top, left, H, W)
    return out, pad_mask, pad_info


def crop_from_square(x: torch.Tensor, pad_info: tuple) -> torch.Tensor:
    top, left, H, W = pad_info
    return x[:, :, top:top + H, left:left + W]


class LearnedFrequencyResponse(nn.Module):
    def __init__(self, hidden: int = 32, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )

    def forward(self, Y: torch.Tensor, Y_hat: torch.Tensor = None) -> torch.Tensor:
        if Y_hat is None:
            Y_hat = rfft2c(Y.float())
        real = Y_hat.real.mean(dim=1, keepdim=True)
        imag = Y_hat.imag.mean(dim=1, keepdim=True)
        mag = Y_hat.abs().mean(dim=1, keepdim=True)
        feat = torch.cat([real, imag, mag], dim=1)
        out = self.net(feat)
        real_out = out[:, 0:1, :, :] + 1.0
        imag_out = out[:, 1:2, :, :]
        return torch.complex(real_out, imag_out)


class LearnedSigmaEff(nn.Module):
    def __init__(self, hidden: int = 32, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, H_hat: torch.Tensor, Y: torch.Tensor, Y_hat: torch.Tensor = None) -> torch.Tensor:
        if Y_hat is None:
            Y_hat = rfft2c(Y.float())
        h_mag = H_hat.abs()
        y_mag = Y_hat.abs().mean(dim=1, keepdim=True)
        feat = torch.cat([h_mag, y_mag], dim=1)
        raw = self.net(feat)
        return self.sigma_min + F.softplus(raw)


@dataclass
class SIConfig:
    tau: float = 0.1
    sigma_min: float = 1e-4
    eps: float = 1e-6
    pad_factor: int = 16
    fft_norm: str = "ortho"
    use_y_as_cond: bool = True
    force_residual: bool = True
    h_mode: str = "learned"
    sigma_mode: str = "learned"
    sigma_k: float = 1.0


class SelectiveInversionUformer(nn.Module):
    def __init__(self,
                 uformer: nn.Module,
                 cfg: SIConfig = SIConfig(),
                 h_estimator: nn.Module = None,
                 sigma_estimator: nn.Module = None):
        super().__init__()
        self.uformer = uformer
        self.cfg = cfg
        self.h_estimator = h_estimator or LearnedFrequencyResponse()
        self.sigma_estimator = sigma_estimator or LearnedSigmaEff(sigma_min=cfg.sigma_min)

    def estimate_frequency_response(self, Y: torch.Tensor, Y_hat: torch.Tensor, op_meta: dict) -> torch.Tensor:
        mode = op_meta.get("mode", self.cfg.h_mode)
        if mode == "known":
            H_hat = op_meta["H_hat"]
            if not torch.is_complex(H_hat):
                H_hat = torch.complex(H_hat, torch.zeros_like(H_hat))
            if H_hat.dim() == 3:
                H_hat = H_hat.unsqueeze(0)
            if H_hat.shape[0] == 1 and Y_hat.shape[0] > 1:
                H_hat = H_hat.expand(Y_hat.shape[0], -1, -1, -1)
            return H_hat
        if mode == "identity":
            shape = (Y_hat.shape[0], 1, Y_hat.shape[2], Y_hat.shape[3])
            return torch.ones(shape, device=Y_hat.device, dtype=Y_hat.dtype)
        return self.h_estimator(Y, Y_hat)

    def compute_sigma_eff(self, H_hat: torch.Tensor, Y: torch.Tensor, Y_hat: torch.Tensor, op_meta: dict) -> torch.Tensor:
        mode = op_meta.get("sigma_mode", self.cfg.sigma_mode)
        if mode == "known":
            return op_meta["sigma_eff"]
        if mode == "analytic":
            k = op_meta.get("sigma_k", self.cfg.sigma_k)
            return torch.clamp(k / (H_hat.abs() + self.cfg.eps), min=self.cfg.sigma_min)
        return self.sigma_estimator(H_hat, Y, Y_hat)

    def forward(self, Y: torch.Tensor, op_meta: dict = None) -> dict:
        if op_meta is None:
            op_meta = {}

        Y0 = Y
        Y, pad_mask, pad_info = pad_to_square(Y, factor=self.cfg.pad_factor)
        B, C, H, W = Y.shape

        Y_fp32 = Y.float()
        Y_hat = rfft2c(Y_fp32, norm=self.cfg.fft_norm)
        H_hat = self.estimate_frequency_response(Y_fp32, Y_hat, op_meta)
        sigma_eff = self.compute_sigma_eff(H_hat, Y_fp32, Y_hat, op_meta)

        M_s = (sigma_eff <= self.cfg.tau).float()
        M_n = 1.0 - M_s

        Hc = H_hat.expand(-1, C, -1, -1)
        Xinv_hat = (Y_hat * M_s) * torch.conj(Hc) / (Hc.abs() ** 2 + self.cfg.eps)
        x_ctx = irfft2c(Xinv_hat, s=(H, W), norm=self.cfg.fft_norm)

        x_ctx_in = x_ctx.to(Y.dtype)
        if self.cfg.use_y_as_cond:
            G_in = torch.cat([Y, x_ctx_in], dim=1)
        else:
            G_in = x_ctx_in

        r = self.uformer(G_in)
        if self.cfg.force_residual and getattr(self.uformer, "dd_in", None) == 3:
            r = r - G_in
        r_hat = rfft2c(r.float(), norm=self.cfg.fft_norm) * M_n

        Xrec_hat = Xinv_hat + r_hat
        X_rec = irfft2c(Xrec_hat, s=(H, W), norm=self.cfg.fft_norm).to(Y.dtype)

        X_rec = crop_from_square(X_rec, pad_info)
        x_ctx = crop_from_square(x_ctx, pad_info)
        r = crop_from_square(r, pad_info)

        return {
            "X_rec": X_rec,
            "Xinv_hat": Xinv_hat,
            "r_hat": r_hat,
            "M_s": M_s,
            "M_n": M_n,
            "x_ctx": x_ctx,
            "r": r,
            "H_hat": H_hat,
            "sigma_eff": sigma_eff,
            "pad_mask": pad_mask,
            "input_shape": Y0.shape,
        }
