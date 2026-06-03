# ── training_per_class.py ─────────────────────────────────────────────────────
# Trains one independent reconstruction model per ModelNet10 class.
# Each model only ever sees samples from its own class.
# Ground-truth labels are used for routing — no classifier involved here.
#
# Saves: ./checkpoints_recon/model_<classname>.pth  (10 files total)
#
# Quick-start
# -----------
#   # Train all 10 classes sequentially (safe on any GPU size)
#   python training_per_class.py
#
#   # Train a single class (useful for debugging or resuming)
#   python training_per_class.py --classes chair toilet

import os, time, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from modelnet10_mv_dataset import MultiViewModelNet10Dataset, ToFloat

# ── Config ─────────────────────────────────────────────────────────────────────
CACHE_ROOT   = "./ModelNet10_mv_cache"
SAVE_DIR     = "./checkpoints_recon"
BATCH_SIZE   = 16
NUM_WORKERS  = 4
NUM_EPOCHS   = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
IN_CHANNELS  = 4
N_VIEWS      = 6
LATENT_DIM   = 1024
VOXEL_SIZE   = 32
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ── Architecture (identical to training_multiview.py) ──────────────────────────

class Encoder(nn.Module):
    def _block(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def __init__(self, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels,  32, stride=2),
            self._block(32,           64, stride=2),
            self._block(64,          128, stride=2),
            self._block(128,         256, stride=2),
            self._block(256,         512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, latent_dim)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.fc(self.pool(self.features(x)).flatten(1)))


class Decoder(nn.Module):
    def _deconv_block(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, 256 * 2 * 2 * 2)
        self.network = nn.Sequential(
            self._deconv_block(256, 128),
            self._deconv_block(128,  64),
            self._deconv_block( 64,  32),
            self._deconv_block( 32,  16),
            nn.ConvTranspose3d(16, 1, 1),
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
        return self.conv_out(r2)


class ReconstructionModel(nn.Module):
    """Single-class reconstruction model: (B,N,4,H,W) → coarse + refined logits."""
    def __init__(self, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.refiner = Refiner()

    def forward(self, x):
        B, N, C, H, W = x.shape
        feats   = self.encoder(x.view(B * N, C, H, W)).view(B, N, -1)
        z       = feats.max(dim=1).values
        coarse  = self.decoder(z)
        refined = self.refiner(coarse)
        return coarse, refined


# ── Helpers ────────────────────────────────────────────────────────────────────

def prepare_batch(batch, device):
    rgb    = batch["rgb"].to(device, non_blocking=True)
    depth  = batch["depth"].to(device, non_blocking=True)
    voxels = batch["voxels"].to(device, non_blocking=True)
    rgbd   = torch.cat([rgb, depth], dim=2)     # (B,6,4,H,W)
    if voxels.shape[-1] != VOXEL_SIZE:
        voxels = F.interpolate(voxels, size=(VOXEL_SIZE,)*3, mode="trilinear", align_corners=False)
    return rgbd, (voxels > 0.5).float()


@torch.no_grad()
def voxel_iou(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * target).flatten(1).sum(1)
    union = (pred + target).clamp(max=1).flatten(1).sum(1)
    return (inter / (union + 1e-8)).mean().item()


def make_class_loaders(class_name: str):
    """Build train/test DataLoaders filtered to a single class."""
    label_idx = CLASS_TO_IDX[class_name]
    transform = ToFloat()

    train_ds = MultiViewModelNet10Dataset(
        f"{CACHE_ROOT}/train",
        transform=transform,
        label_filter=[label_idx],
    )
    test_ds = MultiViewModelNet10Dataset(
        f"{CACHE_ROOT}/test",
        transform=transform,
        label_filter=[label_idx],
    )

    loader_kw = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS > 0,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kw)
    test_loader  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **loader_kw)
    return train_loader, test_loader


def train_one_class(class_name: str) -> float:
    """
    Train a reconstruction model for a single class.
    Returns the best validation IoU achieved.
    """
    ckpt_path = os.path.join(SAVE_DIR, f"model_{class_name}.pth")
    if os.path.exists(ckpt_path):
        print(f"  [skip] {ckpt_path} already exists — delete to retrain")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return ckpt.get("val_iou", 0.0)

    print(f"\n{'='*60}")
    print(f"  Training class: {class_name.upper()}")
    print(f"{'='*60}")

    train_loader, test_loader = make_class_loaders(class_name)
    print(f"  Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")

    _model_raw = ReconstructionModel().to(DEVICE)
    model      = torch.compile(_model_raw) if hasattr(torch, "compile") else _model_raw

    pos_weight = torch.tensor([2.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(_model_raw.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)
    amp_on     = DEVICE == "cuda"
    scaler     = torch.amp.GradScaler("cuda", enabled=amp_on)

    history      = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}
    best_val_iou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        tr_loss = tr_iou = 0.0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch:03d}/{NUM_EPOCHS} [train]", leave=False):
            rgbd, voxels = prepare_batch(batch, DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                coarse, refined = model(rgbd)
                loss = 0.3 * criterion(coarse, voxels) + 0.7 * criterion(refined, voxels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_model_raw.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item()
            tr_iou  += voxel_iou(refined, voxels)

        tr_loss /= len(train_loader)
        tr_iou  /= len(train_loader)

        # Validate
        model.eval()
        va_loss = va_iou = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  Epoch {epoch:03d}/{NUM_EPOCHS} [val]  ", leave=False):
                rgbd, voxels = prepare_batch(batch, DEVICE)
                with torch.amp.autocast("cuda", enabled=amp_on):
                    _, refined = model(rgbd)
                va_loss += criterion(refined, voxels).item()
                va_iou  += voxel_iou(refined, voxels)

        va_loss /= len(test_loader)
        va_iou  /= len(test_loader)
        scheduler.step()

        print(
            f"  Epoch {epoch:03d}/{NUM_EPOCHS}  "
            f"train loss={tr_loss:.4f} IoU={tr_iou:.4f}  |  "
            f"val loss={va_loss:.4f} IoU={va_iou:.4f}  "
            f"[{time.time()-t0:.1f}s]"
        )
        for k, v in zip(["train_loss","train_iou","val_loss","val_iou"],
                         [tr_loss, tr_iou, va_loss, va_iou]):
            history[k].append(v)

        if va_iou > best_val_iou:
            best_val_iou = va_iou
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": _model_raw.state_dict(),
                    "val_iou":     va_iou,
                    "val_loss":    va_loss,
                    "class_name":  class_name,
                    "config": {
                        "in_channels": IN_CHANNELS,
                        "latent_dim":  LATENT_DIM,
                        "voxel_size":  VOXEL_SIZE,
                    },
                },
                ckpt_path,
            )
            print(f"  ✓ Saved {ckpt_path}  (val IoU={best_val_iou:.4f})")

    # Per-class training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    ep = range(1, NUM_EPOCHS + 1)
    ax1.plot(ep, history["train_loss"], label="Train"); ax1.plot(ep, history["val_loss"], label="Val")
    ax1.set_title(f"{class_name} — Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(ep, history["train_iou"],  label="Train"); ax2.plot(ep, history["val_iou"],  label="Val")
    ax2.set_title(f"{class_name} — IoU"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"curves_{class_name}.png"), dpi=120, bbox_inches="tight")

    return best_val_iou


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes", nargs="+", default=CLASSES, choices=CLASSES,
        help="Which classes to train (default: all 10)",
    )
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device  : {DEVICE}")
    print(f"Classes : {args.classes}")
    print(f"Output  : {SAVE_DIR}\n")

    results = {}
    for class_name in args.classes:
        results[class_name] = train_one_class(class_name)

    print(f"\n{'='*60}")
    print("  Final best val IoU per class")
    print(f"{'='*60}")
    for cls, iou in results.items():
        print(f"  {cls:<14}  {iou:.4f}")
    print(f"\n  Mean IoU: {sum(results.values())/len(results):.4f}")
