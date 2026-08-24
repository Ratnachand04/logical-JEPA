"""Convolutional autoencoder baseline.

The classical unsupervised anomaly detector: train a bottlenecked autoencoder on
normal images, then flag pixels it reconstructs badly.

It is the right baseline to contrast against JEPA because it fails for a
*specific, predictable* reason. An autoencoder is trained to copy its input, and
convolutions are local, so a plausible-looking screw in the wrong place is
reconstructed perfectly well -- the network never has to know where screws
belong. It therefore detects structural anomalies (unfamiliar texture) but is
close to blind to logical ones.

That is exactly the gap the JEPA formulation is meant to close: predicting a
*hidden* region from its context is a fundamentally different task from copying
a visible one.

Trained from scratch, no pretrained weights, consistent with the rest of the
project.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """Symmetric conv encoder/decoder with a low-dimensional bottleneck.

    Args:
        in_chans: image channels.
        base: channel width of the first stage; doubles each downsample.
        latent_dim: bottleneck width. Deliberately small -- a wide bottleneck
            lets the model copy anomalies straight through, which is the
            standard failure mode of AE-based detection.
        img_size: input resolution (must be divisible by 32).
    """

    def __init__(self, in_chans: int = 3, base: int = 32, latent_dim: int = 128,
                 img_size: int = 256):
        super().__init__()
        if img_size % 32 != 0:
            raise ValueError("img_size must be divisible by 32")

        self.img_size = img_size
        self.bottleneck_size = img_size // 32   # 5 downsamples

        def down(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def up(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            down(in_chans, base),        # 128
            down(base, base * 2),        # 64
            down(base * 2, base * 4),    # 32
            down(base * 4, base * 8),    # 16
            down(base * 8, base * 8),    # 8
            nn.Conv2d(base * 8, latent_dim, 3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            up(base * 8, base * 8),      # 16
            up(base * 8, base * 4),      # 32
            up(base * 4, base * 2),      # 64
            up(base * 2, base),          # 128
            nn.ConvTranspose2d(base, in_chans, 4, stride=2, padding=1),
            nn.Tanh(),                   # matches the [-1, 1] normalisation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        return f"ConvAutoencoder({self.num_parameters()/1e6:.2f}M params)"


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor,
                        l1_weight: float = 0.5) -> torch.Tensor:
    """MSE plus L1. The L1 term sharpens edges that pure MSE blurs away."""
    return F.mse_loss(recon, target) + l1_weight * F.l1_loss(recon, target)


class AutoencoderScorer:
    """Anomaly scoring by reconstruction error, matching AnomalyScorer's API.

    The error map is the per-pixel channel-mean squared difference, blurred with
    the same Gaussian the JEPA maps use so the two are compared on equal terms.
    """

    def __init__(self, model: ConvAutoencoder, sigma: float = 4.0,
                 aggregation: str = "topk", top_k_ratio: float = 0.01):
        self.model = model
        self.sigma = sigma
        self.aggregation = aggregation
        self.top_k_ratio = top_k_ratio

        from anomaly.scoring import Calibration
        self.calibration = Calibration()

    @torch.no_grad()
    def score_batch(self, images: torch.Tensor) -> dict:
        from anomaly.anomaly_map import gaussian_blur
        from anomaly.scoring import aggregate_score

        self.model.eval()
        recon = self.model(images)

        error = ((images - recon) ** 2).mean(dim=1, keepdim=True)
        maps = gaussian_blur(error, self.sigma)
        scores = aggregate_score(maps, self.aggregation, self.top_k_ratio)

        return {"scores": scores, "maps": maps, "recon": recon}

    @torch.no_grad()
    def score_loader(self, loader, device, collect_maps: bool = True,
                     progress: bool = False) -> dict:
        import numpy as np

        iterator = loader
        if progress:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="AE scoring", leave=False)

        scores, labels, defects, maps, masks = [], [], [], [], []
        for batch in iterator:
            images = batch["image"].to(device, non_blocking=True)
            out = self.score_batch(images)

            scores.append(out["scores"].float().cpu().numpy())
            labels.append(batch["label"].numpy())
            defects.extend(batch["defect_type"])
            if collect_maps:
                maps.append(out["maps"].float().cpu().numpy())
                if "mask" in batch:
                    masks.append(batch["mask"].float().cpu().numpy())

        result = {
            "scores": np.concatenate(scores) if scores else np.array([]),
            "labels": np.concatenate(labels) if labels else np.array([]),
            "defect_types": np.array(defects),
        }
        if collect_maps and maps:
            result["maps"] = np.concatenate(maps)
            if masks:
                result["masks"] = np.concatenate(masks)
        return result

    @torch.no_grad()
    def calibrate(self, loader, device, sigma_threshold: float = 3.0):
        from anomaly.scoring import fit_calibration

        out = self.score_loader(loader, device, collect_maps=True)
        self.calibration = fit_calibration(
            out["scores"], out.get("maps"), sigma_threshold=sigma_threshold
        )
        return self.calibration


def train_autoencoder(cfg, train_loader, device, logger, epochs: int | None = None
                      ) -> ConvAutoencoder:
    """Train the AE baseline on normal images only.

    Uses the same optimiser family, schedule shape and epoch budget as the JEPA
    run so the comparison reflects the objective, not the training budget.
    """
    import math

    epochs = epochs or cfg.get_path("train.epochs", 150)
    img_size = cfg.get_path("data.img_size", 256)

    model = ConvAutoencoder(
        base=cfg.get_path("baseline.ae_base", 32),
        latent_dim=cfg.get_path("baseline.ae_latent", 128),
        img_size=img_size,
    ).to(device)
    logger.info(model.describe())

    base_lr = cfg.get_path("baseline.ae_lr", 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    total_steps = max(len(train_loader) * epochs, 1)
    step = 0

    from tqdm import tqdm

    for epoch in range(epochs):
        model.train()
        running, count = 0.0, 0

        for batch in tqdm(train_loader, desc=f"AE epoch {epoch+1}/{epochs}", leave=False):
            images = batch["image"].to(device, non_blocking=True)

            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
            for group in optimizer.param_groups:
                group["lr"] = lr

            recon = model(images)
            loss = reconstruction_loss(recon, images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.detach())
            count += 1
            step += 1

        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            logger.info(f"AE epoch {epoch+1}/{epochs} recon_loss={running/max(count,1):.5f}")

    return model
