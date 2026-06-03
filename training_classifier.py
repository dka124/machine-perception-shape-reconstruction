# ── training_classifier.py ────────────────────────────────────────────────────
# Trains a multi-view RGB-D → class classifier over all 10 ModelNet10 classes.
# Saves:  ./checkpoints_cls/classifier.pth
#
# Run BEFORE training_per_class.py.
#
# Quick-start
# -----------
#   python training_classifier.py

import os, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from modelnet10_mv_dataset import make_mv_dataloaders, ToFloat

# ── Config ─────────────────────────────────────────────────────────────────────
CACHE_ROOT   = "./ModelNet10_mv_cache"
SAVE_DIR     = "./checkpoints_cls"
BATCH_SIZE   = 32
NUM_WORKERS  = 8
NUM_EPOCHS   = 40
LR           = 1e-3
WEIGHT_DECAY = 1e-4
IN_CHANNELS  = 4        # RGB (3) + Depth (1)
N_VIEWS      = 6
LATENT_DIM   = 512      # smaller than reconstruction — classification needs less capacity
NUM_CLASSES  = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]


# ── Architecture ───────────────────────────────────────────────────────────────

class ViewEncoder(nn.Module):
    """Shared-weight 2-D CNN: (B, 4, H, W) → (B, LATENT_DIM)."""
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


class MultiViewClassifier(nn.Module):
    """
    (B, N_VIEWS, 4, H, W)  →  class logits  (B, NUM_CLASSES)

    Each view is encoded independently by the shared ViewEncoder,
    then fused with element-wise max-pooling (permutation-invariant),
    then classified by a small MLP head.
    """
    def __init__(self, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = ViewEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.head    = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x : (B, N, C, H, W)
        B, N, C, H, W = x.shape
        feats = self.encoder(x.view(B * N, C, H, W)).view(B, N, -1)  # (B,N,D)
        z     = feats.max(dim=1).values                                # (B,D)
        return self.head(z)                                            # (B, num_classes)


# ── Helpers ────────────────────────────────────────────────────────────────────

def prepare_batch(batch, device):
    rgb   = batch["rgb"].to(device, non_blocking=True)    # (B,6,3,H,W)
    depth = batch["depth"].to(device, non_blocking=True)  # (B,6,1,H,W)
    label = batch["label"].to(device, non_blocking=True)  # (B,)
    rgbd  = torch.cat([rgb, depth], dim=2)                # (B,6,4,H,W)
    return rgbd, label


@torch.no_grad()
def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")

    train_loader, test_loader = make_mv_dataloaders(
        CACHE_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        prefetch_factor=4, persistent_workers=True,
    )
    print(f"Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}\n")

    _model_raw = MultiViewClassifier().to(DEVICE)
    model      = torch.compile(_model_raw) if hasattr(torch, "compile") else _model_raw
    print(f"Params: {sum(p.numel() for p in _model_raw.parameters() if p.requires_grad):,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(_model_raw.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)
    amp_on    = DEVICE == "cuda"
    scaler    = torch.amp.GradScaler("cuda", enabled=amp_on)

    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        tr_loss = tr_acc = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{NUM_EPOCHS} [train]", leave=False):
            rgbd, labels = prepare_batch(batch, DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = model(rgbd)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_model_raw.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item()
            tr_acc  += accuracy(logits, labels)

        tr_loss /= len(train_loader)
        tr_acc  /= len(train_loader)

        # Validate
        model.eval()
        va_loss = va_acc = 0.0
        # Per-class confusion tracking
        correct_per_class = torch.zeros(NUM_CLASSES)
        total_per_class   = torch.zeros(NUM_CLASSES)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch:03d}/{NUM_EPOCHS} [val]  ", leave=False):
                rgbd, labels = prepare_batch(batch, DEVICE)
                with torch.amp.autocast("cuda", enabled=amp_on):
                    logits = model(rgbd)
                va_loss += criterion(logits, labels).item()
                va_acc  += accuracy(logits, labels)
                preds = logits.argmax(dim=1).cpu()
                for c in range(NUM_CLASSES):
                    mask = (labels.cpu() == c)
                    total_per_class[c]   += mask.sum()
                    correct_per_class[c] += (preds[mask] == c).sum()

        va_loss /= len(test_loader)
        va_acc  /= len(test_loader)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
            f"train loss={tr_loss:.4f}  acc={tr_acc:.3f}  |  "
            f"val loss={va_loss:.4f}  acc={va_acc:.3f}  "
            f"[{time.time()-t0:.1f}s]  lr={scheduler.get_last_lr()[0]:.2e}"
        )
        for history_key, value in zip(
            ["train_loss", "train_acc", "val_loss", "val_acc"],
            [tr_loss, tr_acc, va_loss, va_acc],
        ):
            history[history_key].append(value)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": _model_raw.state_dict(),
                    "val_acc":     va_acc,
                    "val_loss":    va_loss,
                    "config": {
                        "in_channels": IN_CHANNELS,
                        "latent_dim":  LATENT_DIM,
                        "num_classes": NUM_CLASSES,
                    },
                    "classes": CLASSES,
                },
                os.path.join(SAVE_DIR, "classifier.pth"),
            )
            print(f"  ✓ Saved classifier  (val acc={best_val_acc:.3f})")

    print(f"\nTraining complete.  Best val accuracy: {best_val_acc:.3f}")

    # Per-class accuracy at best epoch (reloaded)
    print("\nPer-class accuracy (last epoch):")
    for i, cls in enumerate(CLASSES):
        acc_c = (correct_per_class[i] / total_per_class[i].clamp(min=1)).item()
        print(f"  {cls:<12} {acc_c*100:5.1f}%")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, NUM_EPOCHS + 1)
    ax1.plot(ep, history["train_loss"], label="Train"); ax1.plot(ep, history["val_loss"], label="Val")
    ax1.set_title("Cross-Entropy Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
    ax2.plot(ep, history["train_acc"],  label="Train"); ax2.plot(ep, history["val_acc"],  label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves_classifier.png", dpi=150, bbox_inches="tight")
    print("Saved training_curves_classifier.png")
