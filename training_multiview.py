# %% [markdown]
# # Multi-View RGB-D → Voxel Reconstruction (Pix2Vox-style)
#
# **Architecture:** Encoder → View Pooling → Decoder → Refiner
# **Input:** 6 RGB-D images per object (4 sides + top + bottom)
# **Output:** 32³ voxel occupancy grid
# **Dataset:** ModelNet10 multi-view cache (one .pt per object)
#
# ## How multi-view fusion works
#
# Each of the 6 views is encoded independently by the shared Encoder
# (weight-sharing means any single view could also work at inference).
# The 6 feature vectors are fused by element-wise MAX pooling — this is
# the same approach used in Pix2Vox and PointNet: the max operation is
# permutation-invariant and naturally picks the most "informative" signal
# from any view for each feature dimension.
#
#   views (B,6,4,H,W) → encoder → (B,6,D) → max over views → (B,D) → decoder
#
# ## Quick-start
# ```bash
# # Step 1 — pre-render once (~10–30 min depending on hardware)
# python preprocess_multiview.py \
#     --root ./ModelNet10 --cache ./ModelNet10_mv_cache --workers 8
#
# # Step 2 — train
# python training_multiview.py
# ```

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from modelnet10_mv_dataset import make_mv_dataloaders, MultiViewModelNet10Dataset, ToFloat

# ── Config ─────────────────────────────────────────────────────────────────────
CACHE_ROOT   = "./ModelNet10_mv_cache"
BATCH_SIZE   = 16     # each item now holds 6 views; reduce if VRAM is tight
NUM_WORKERS  = 8
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
VOXEL_SIZE   = 32
IN_CHANNELS  = 4      # RGB (3) + Depth (1)
N_VIEWS      = 6
LATENT_DIM   = 1024
NUM_EPOCHS   = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SAVE_DIR     = "./checkpoints_mv"


# ── Model ──────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    2-D CNN that maps a single (B, 4, H, W) RGB-D image → (B, LATENT_DIM) vector.
    Shared across all views (same weights).
    """
    def _conv_block(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def __init__(self, in_channels: int = IN_CHANNELS, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(in_channels,  32, stride=2),
            self._conv_block(32,           64, stride=2),
            self._conv_block(64,          128, stride=2),
            self._conv_block(128,         256, stride=2),
            self._conv_block(256,         512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, latent_dim)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.act(self.fc(x))          # (B, LATENT_DIM)


class ViewPooling(nn.Module):
    """
    Fuse N view features into one via element-wise MAX pooling.
    Permutation-invariant — order of views does not matter.
    """
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features : (B, N_VIEWS, LATENT_DIM)
        return features.max(dim=1).values    # (B, LATENT_DIM)


class Decoder(nn.Module):
    """Latent vector → coarse 32³ voxel grid (logits)."""
    def _deconv_block(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, 256 * 2 * 2 * 2)
        self.network = nn.Sequential(
            self._deconv_block(256, 128),   # 2→4
            self._deconv_block(128,  64),   # 4→8
            self._deconv_block( 64,  32),   # 8→16
            self._deconv_block( 32,  16),   # 16→32
            nn.ConvTranspose3d(16, 1, 1),   # logits
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(self.fc(z).view(-1, 256, 2, 2, 2))   # (B,1,32,32,32)


class Refiner(nn.Module):
    """Residual 3-D CNN that sharpens the coarse prediction."""
    def _res_block(self, ch):
        return nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
        )

    def __init__(self):
        super().__init__()
        self.conv_in  = nn.Conv3d(1, 32, 3, padding=1)
        self.res1     = self._res_block(32)
        self.res2     = self._res_block(32)
        self.conv_out = nn.Conv3d(32, 1, 1)
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        x  = self.relu(self.conv_in(coarse))
        r1 = self.relu(self.res1(x) + x)
        r2 = self.relu(self.res2(r1) + r1)
        return self.conv_out(r2)   # logits


class MultiViewRGBDToVoxel(nn.Module):
    """
    Full pipeline:
        (B, N, 4, H, W)  →  encoder per view  →  max pool  →  decoder  →  refiner
        returns (coarse_logits, refined_logits)  both (B, 1, 32, 32, 32)
    """
    def __init__(self, in_channels: int = IN_CHANNELS, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder  = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.pooling  = ViewPooling()
        self.decoder  = Decoder(latent_dim=latent_dim)
        self.refiner  = Refiner()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x : (B, N_VIEWS, C, H, W)
        B, N, C, H, W = x.shape

        # Encode all views with the shared encoder
        # Flatten B and N so the 2-D CNN sees (B*N, C, H, W)
        x_flat    = x.view(B * N, C, H, W)
        feat_flat = self.encoder(x_flat)             # (B*N, LATENT_DIM)
        features  = feat_flat.view(B, N, -1)         # (B, N, LATENT_DIM)

        # Fuse views
        z = self.pooling(features)                   # (B, LATENT_DIM)

        # Decode
        coarse   = self.decoder(z)                   # (B, 1, 32, 32, 32)
        refined  = self.refiner(coarse)              # (B, 1, 32, 32, 32)
        return coarse, refined


# ── Helpers ────────────────────────────────────────────────────────────────────

def prepare_batch(batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack a DataLoader batch and build the RGBD tensor.

    Input tensors from DataLoader:
        rgb   : (B, 6, 3, H, W)  float32
        depth : (B, 6, 1, H, W)  float32
    Combined:
        rgbd  : (B, 6, 4, H, W)  float32  ← model input

    Returns (rgbd, voxels_binary).
    """
    rgb    = batch["rgb"].to(device, non_blocking=True)     # (B,6,3,H,W)
    depth  = batch["depth"].to(device, non_blocking=True)   # (B,6,1,H,W)
    voxels = batch["voxels"].to(device, non_blocking=True)  # (B,1,R,R,R)

    rgbd = torch.cat([rgb, depth], dim=2)                   # (B,6,4,H,W)

    if voxels.shape[-1] != VOXEL_SIZE:
        voxels = F.interpolate(
            voxels, size=(VOXEL_SIZE,) * 3,
            mode="trilinear", align_corners=False,
        )
    return rgbd, (voxels > 0.5).float()


@torch.no_grad()
def voxel_iou(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred_logits) > threshold).float()
    inter    = (pred_bin * target).flatten(1).sum(1)
    union    = (pred_bin + target).clamp(max=1).flatten(1).sum(1)
    return (inter / (union + 1e-8)).mean().item()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = make_mv_dataloaders(
        CACHE_ROOT,
        batch_size         = BATCH_SIZE,
        num_workers        = NUM_WORKERS,
        prefetch_factor    = 4,
        persistent_workers = True,
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    # Sanity-check batch shapes
    _b = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  rgb    {tuple(_b['rgb'].shape)}   dtype={_b['rgb'].dtype}")
    print(f"  depth  {tuple(_b['depth'].shape)}  dtype={_b['depth'].dtype}")
    print(f"  voxels {tuple(_b['voxels'].shape)}  dtype={_b['voxels'].dtype}")
    del _b

    # ── Model ─────────────────────────────────────────────────────────────────
    _model_raw = MultiViewRGBDToVoxel(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(DEVICE)
    model      = torch.compile(_model_raw) if hasattr(torch, "compile") else _model_raw
    n_params   = sum(p.numel() for p in _model_raw.parameters() if p.requires_grad)
    print(f"\nTrainable params : {n_params:,}")
    print(f"torch.compile    : {'enabled' if hasattr(torch, 'compile') else 'disabled'}")

    # ── Loss / optimiser / AMP ────────────────────────────────────────────────
    pos_weight = torch.tensor([2.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(_model_raw.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)
    amp_on     = DEVICE == "cuda"
    scaler     = torch.amp.GradScaler("cuda", enabled=amp_on)
    print(f"AMP              : {'enabled' if amp_on else 'disabled'}\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    history      = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}
    best_val_iou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = train_iou = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{NUM_EPOCHS} [train]", leave=False):
            rgbd, voxels = prepare_batch(batch, DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                coarse, refined = model(rgbd)
                # Weighted sum: coarse loss guides early training,
                # refined loss is the primary objective
                loss = 0.3 * criterion(coarse, voxels) + 0.7 * criterion(refined, voxels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_model_raw.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_iou  += voxel_iou(refined, voxels)

        train_loss /= len(train_loader)
        train_iou  /= len(train_loader)

        # Validate
        model.eval()
        val_loss = val_iou = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch:03d}/{NUM_EPOCHS} [val]  ", leave=False):
                rgbd, voxels = prepare_batch(batch, DEVICE)
                with torch.amp.autocast("cuda", enabled=amp_on):
                    _, refined = model(rgbd)
                val_loss += criterion(refined, voxels).item()
                val_iou  += voxel_iou(refined, voxels)

        val_loss /= len(test_loader)
        val_iou  /= len(test_loader)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
            f"train loss={train_loss:.4f}  IoU={train_iou:.4f}  |  "
            f"val loss={val_loss:.4f}  IoU={val_iou:.4f}  "
            f"[{elapsed:.1f}s]  lr={scheduler.get_last_lr()[0]:.2e}"
        )
        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": _model_raw.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_iou":     val_iou,
                    "val_loss":    val_loss,
                    "config": {
                        "in_channels": IN_CHANNELS,
                        "latent_dim":  LATENT_DIM,
                        "n_views":     N_VIEWS,
                        "voxel_size":  VOXEL_SIZE,
                    },
                },
                os.path.join(SAVE_DIR, "best_model.pth"),
            )
            print(f"  ✓ Saved best model  (val IoU={best_val_iou:.4f})")

    print(f"\nTraining complete.  Best val IoU: {best_val_iou:.4f}")

    # ── Training curves ────────────────────────────────────────────────────────
    epochs_x = range(1, NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(epochs_x, history["train_loss"], label="Train")
    ax1.plot(epochs_x, history["val_loss"],   label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Weighted BCE Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs_x, history["train_iou"], label="Train")
    ax2.plot(epochs_x, history["val_iou"],   label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU")
    ax2.set_title("IoU"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves_mv.png", dpi=150, bbox_inches="tight")
    print("Saved training_curves_mv.png")

    # ── Qualitative visualisation ──────────────────────────────────────────────
    ckpt = torch.load(
        os.path.join(SAVE_DIR, "best_model.pth"),
        map_location=DEVICE, weights_only=True,
    )
    _model_raw.load_state_dict(ckpt["model_state"])
    model = torch.compile(_model_raw) if hasattr(torch, "compile") else _model_raw
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val IoU={ckpt['val_iou']:.4f}")

    batch          = next(iter(test_loader))
    rgbd, voxel_gt = prepare_batch(batch, DEVICE)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp_on):
            _, refined = model(rgbd)
            probs      = torch.sigmoid(refined)

    idx    = 0
    gt_np  = voxel_gt[idx, 0].cpu().numpy()
    pd_np  = (probs[idx, 0] > 0.5).float().cpu().numpy()
    slices = [VOXEL_SIZE // 4, VOXEL_SIZE // 2, 3 * VOXEL_SIZE // 4]

    # Show 3 of the 6 input views (front, top, side)
    view_labels = ["front (0°)", "right (90°)", "back (180°)",
                   "left (270°)", "top", "bottom"]
    show_views  = [0, 4, 1]   # front, top, right

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for col, vi in enumerate(show_views):
        rgb_np = batch["rgb"][idx, vi].permute(1, 2, 0).numpy()
        axes[0, col].imshow(rgb_np)
        axes[0, col].set_title(f"Input: {view_labels[vi]}", fontsize=9)
        axes[0, col].axis("off")

    for col, sl in enumerate(slices):
        axes[1, col].imshow(gt_np[sl], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"GT z={sl}"); axes[1, col].axis("off")
        axes[2, col].imshow(pd_np[sl], cmap="gray", vmin=0, vmax=1)
        axes[2, col].set_title(f"Pred z={sl}"); axes[2, col].axis("off")

    sample_iou = voxel_iou(refined[idx:idx+1], voxel_gt[idx:idx+1])
    plt.suptitle(
        f"Label={int(batch['label'][idx])}  |  Sample IoU={sample_iou:.4f}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("qualitative_mv.png", dpi=150, bbox_inches="tight")
    print("Saved qualitative_mv.png")

    # ── Single-object inference example ───────────────────────────────────────
    test_ds = MultiViewModelNet10Dataset(f"{CACHE_ROOT}/test", transform=ToFloat())
    sample  = test_ds[0]

    # Concatenate rgb + depth along channel dim for all 6 views
    rgb_s   = sample["rgb"]    # (6, 3, H, W)
    depth_s = sample["depth"]  # (6, 1, H, W)
    rgbd_s  = torch.cat([rgb_s, depth_s], dim=1).unsqueeze(0).to(DEVICE)  # (1,6,4,H,W)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp_on):
            _, refined = model(rgbd_s)

    vox = (torch.sigmoid(refined[0, 0]) > 0.5).cpu().numpy()
    print(f"\nInference example:")
    print(f"  Input shape     : {tuple(rgbd_s.shape)}")
    print(f"  Output shape    : {vox.shape}")
    print(f"  Occupied voxels : {vox.sum()} / {vox.size}  ({100 * vox.mean():.1f}%)")
    print(f"  Label           : {int(sample['label'])}")
