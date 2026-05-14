# %% [markdown]
# # RGB-D → Voxel Reconstruction (Pix2Vox-style)
# **Architecture:** Encoder → Decoder → Refiner
# **Input:** RGB-D image (4 channels: RGB + depth)
# **Output:** 32³ voxel occupancy grid
# **Dataset:** ModelNet10 — reads pre-rendered `.pt` cache
#
# ## Quick-start
# ```bash
# # Step 1 — pre-render once (~5–15 min with 8 workers)
# pip install open3d trimesh
# python preprocess_dataset.py --root ./ModelNet10 --cache ./ModelNet10_cache --workers 8
#
# # Step 2 — train
# python training.py
# ```

# ── Imports (module level — workers re-import these) ──────────────────────────
import os
import time

import matplotlib

matplotlib.use("Agg")   # non-interactive backend; safe in worker processes
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from modelnet10_cached_dataset import make_cached_dataloaders, CachedModelNet10Dataset, ToFloat

# ── Constants (module level — must be readable by workers) ────────────────────
CACHE_ROOT   = "./ModelNet10_cache"
BATCH_SIZE   = 32
NUM_WORKERS  = 8
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
VOXEL_SIZE   = 32
IN_CHANNELS  = 4
NUM_EPOCHS   = 50
LR           = 1e-3
LR_STEP      = 20
LR_GAMMA     = 0.5
WEIGHT_DECAY = 1e-4
SAVE_DIR     = "./checkpoints"

# ── Model classes (module level — must be picklable by workers) ───────────────

class Encoder(nn.Module):
    def _conv_block(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def __init__(self, in_channels=4, latent_dim=1024):
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

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.act(self.fc(x))


class Decoder(nn.Module):
    def _deconv_block(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def __init__(self, latent_dim=1024):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, 256 * 2 * 2 * 2)
        self.network = nn.Sequential(
            self._deconv_block(256, 128),
            self._deconv_block(128,  64),
            self._deconv_block( 64,  32),
            self._deconv_block( 32,  16),
            nn.ConvTranspose3d(16, 1, 1),   # output logits (no sigmoid)
        )

    def forward(self, z):
        return self.network(self.fc(z).view(-1, 256, 2, 2, 2))


class Refiner(nn.Module):
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

    def forward(self, coarse):
        x  = self.relu(self.conv_in(coarse))
        r1 = self.relu(self.res1(x) + x)
        r2 = self.relu(self.res2(r1) + r1)
        return self.conv_out(r2)   # logits – no sigmoid


class RGBDToVoxel(nn.Module):
    def __init__(self, in_channels=4, latent_dim=1024):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.refiner = Refiner()

    def forward(self, x):
        z      = self.encoder(x)
        coarse = self.decoder(z)
        return coarse, self.refiner(coarse)


# ── Helper functions (module level — picklable, no side effects) ──────────────

def get_rgbd_and_voxel(batch, device):
    rgb    = batch["rgb"].to(device, non_blocking=True)
    depth  = batch["depth"].to(device, non_blocking=True)
    voxels = batch["voxels"].to(device, non_blocking=True)
    rgbd   = torch.cat([rgb, depth], dim=1)
    if voxels.shape[-1] != VOXEL_SIZE:
        voxels = F.interpolate(
            voxels, size=(VOXEL_SIZE,) * 3,
            mode="trilinear", align_corners=False,
        )
    return rgbd, (voxels > 0.5).float()


@torch.no_grad()
def voxel_iou(pred_logits, target, threshold=0.5):
    """Compute IoU from logits (applies sigmoid internally)."""
    pred_probs = torch.sigmoid(pred_logits)
    pred_bin   = (pred_probs > threshold).float()
    inter      = (pred_bin * target).flatten(1).sum(1)
    union      = (pred_bin + target).clamp(max=1).flatten(1).sum(1)
    return (inter / (union + 1e-8)).mean().item()


# ── Everything that EXECUTES goes here — safe on Windows spawn ────────────────
if __name__ == "__main__":

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader, test_loader = make_cached_dataloaders(
        CACHE_ROOT,
        batch_size         = BATCH_SIZE,
        num_workers        = NUM_WORKERS,
        prefetch_factor    = 4,
        persistent_workers = True,
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    # Sanity-check one batch
    _b = next(iter(train_loader))
    print(f"\nBatch shapes after ToFloat():")
    print(f"  rgb    {tuple(_b['rgb'].shape)}  dtype={_b['rgb'].dtype}  range=[{_b['rgb'].min():.2f},{_b['rgb'].max():.2f}]")
    print(f"  depth  {tuple(_b['depth'].shape)}  dtype={_b['depth'].dtype}  range=[{_b['depth'].min():.2f},{_b['depth'].max():.2f}]")
    print(f"  voxels {tuple(_b['voxels'].shape)}  dtype={_b['voxels'].dtype}")
    del _b

    # ── Model ─────────────────────────────────────────────────────────────────
    # _model_for_ckpt: the raw (uncompiled) module used for state_dict + grad clipping
    # model:           the compiled wrapper used in the forward pass
    _model_for_ckpt = RGBDToVoxel(in_channels=IN_CHANNELS, latent_dim=1024).to(DEVICE)
    model = torch.compile(_model_for_ckpt) if hasattr(torch, "compile") else _model_for_ckpt
    print(f"\nTrainable params : {sum(p.numel() for p in _model_for_ckpt.parameters() if p.requires_grad):,}")
    print(f"torch.compile    : {'enabled' if hasattr(torch, 'compile') else 'not available'}")

    # ── Loss / optimiser / AMP ────────────────────────────────────────────────
    # Use BCEWithLogitsLoss (safe for autocast) with positive weight 2.0
    pos_weight_tensor = torch.tensor([2.0]).to(DEVICE)
    criterion         = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer         = optim.Adam(_model_for_ckpt.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler         = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)
    _amp_enabled      = (DEVICE == "cuda")
    scaler            = torch.amp.GradScaler("cuda", enabled=_amp_enabled)
    print(f"AMP              : {'enabled' if _amp_enabled else 'disabled'}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    history      = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}
    best_val_iou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = train_iou = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{NUM_EPOCHS} [train]", leave=False):
            rgbd, voxels = get_rgbd_and_voxel(batch, DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=_amp_enabled):
                coarse_logits, refined_logits = model(rgbd)
                loss = 0.3 * criterion(coarse_logits, voxels) + 0.7 * criterion(refined_logits, voxels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_model_for_ckpt.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_iou  += voxel_iou(refined_logits, voxels)
        train_loss /= len(train_loader)
        train_iou  /= len(train_loader)

        # Validate
        model.eval()
        val_loss = val_iou = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch:03d}/{NUM_EPOCHS} [val]  ", leave=False):
                rgbd, voxels = get_rgbd_and_voxel(batch, DEVICE)
                with torch.amp.autocast("cuda", enabled=_amp_enabled):
                    _, refined_logits = model(rgbd)
                val_loss += criterion(refined_logits, voxels).item()
                val_iou  += voxel_iou(refined_logits, voxels)
        val_loss /= len(test_loader)
        val_iou  /= len(test_loader)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
            f"train loss={train_loss:.4f}  IoU={train_iou:.4f}  |  "
            f"val loss={val_loss:.4f}  IoU={val_iou:.4f}  "
            f"[{time.time()-t0:.1f}s]  lr={scheduler.get_last_lr()[0]:.2e}"
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
                    "model_state": _model_for_ckpt.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_iou":     val_iou,
                    "val_loss":    val_loss,
                    "config":      {"in_channels": IN_CHANNELS, "latent_dim": 1024, "voxel_size": VOXEL_SIZE},
                },
                os.path.join(SAVE_DIR, "best_model.pth"),
            )
            print(f"  ✓ Saved best model  (val IoU={best_val_iou:.4f})")

    print(f"\nTraining complete.  Best val IoU: {best_val_iou:.4f}")

    # ── Training curves ───────────────────────────────────────────────────────
    epochs_x = range(1, NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(epochs_x, history["train_loss"], label="Train")
    ax1.plot(epochs_x, history["val_loss"],   label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Weighted BCE Loss (logits)")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs_x, history["train_iou"], label="Train")
    ax2.plot(epochs_x, history["val_iou"],   label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU")
    ax2.set_title("IoU"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    print("Saved training_curves.png")

    # ── Qualitative visualisation ─────────────────────────────────────────────
    ckpt = torch.load(
        os.path.join(SAVE_DIR, "best_model.pth"),
        map_location=DEVICE, weights_only=True,
    )
    _model_for_ckpt.load_state_dict(ckpt["model_state"])
    model = torch.compile(_model_for_ckpt) if hasattr(torch, "compile") else _model_for_ckpt
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val IoU={ckpt['val_iou']:.4f}")

    batch          = next(iter(test_loader))
    rgbd, voxel_gt = get_rgbd_and_voxel(batch, DEVICE)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=_amp_enabled):
            _, refined_logits = model(rgbd)
            refined_probs = torch.sigmoid(refined_logits)   # convert to [0,1] probabilities

    idx    = 0
    gt_np  = voxel_gt[idx, 0].cpu().numpy()
    pd_np  = (refined_probs[idx, 0] > 0.5).float().cpu().numpy()
    rgb_np = batch["rgb"][idx].permute(1, 2, 0).numpy()
    slices = [VOXEL_SIZE // 4, VOXEL_SIZE // 2, 3 * VOXEL_SIZE // 4]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes[0, 1].imshow(rgb_np)
    axes[0, 1].set_title(f"Input RGB (label={int(batch['label'][idx])})")
    for ax in (axes[0, 0], axes[0, 2], axes[0, 1]):
        ax.axis("off")
    for col, sl in enumerate(slices):
        axes[1, col].imshow(gt_np[sl], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"GT z={sl}"); axes[1, col].axis("off")
        axes[2, col].imshow(pd_np[sl], cmap="gray", vmin=0, vmax=1)
        axes[2, col].set_title(f"Pred z={sl}"); axes[2, col].axis("off")
    plt.suptitle(f"Sample IoU={voxel_iou(refined_logits[idx:idx+1], voxel_gt[idx:idx+1]):.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig("qualitative.png", dpi=150, bbox_inches="tight")
    print("Saved qualitative.png")

    # ── Single-sample inference example ───────────────────────────────────────
    test_ds = CachedModelNet10Dataset(f"{CACHE_ROOT}/test", transform=ToFloat())
    sample  = test_ds[0]
    rgbd_s  = torch.cat([sample["rgb"], sample["depth"]], dim=0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=_amp_enabled):
            _, refined_logits = model(rgbd_s)
    vox = (torch.sigmoid(refined_logits[0, 0]) > 0.5).cpu().numpy()
    print(f"\nInference example:")
    print(f"  Output shape    : {vox.shape}")
    print(f"  Occupied voxels : {vox.sum()} / {vox.size}  ({100*vox.mean():.1f}%)")
    print(f"  Label           : {int(sample['label'])}")