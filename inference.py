# ── inference.py ──────────────────────────────────────────────────────────────
# Full inference pipeline:
#
#   RGB-D input (6 views)
#       │
#       ▼
#   MultiViewClassifier  →  top-1 class prediction
#       │
#       ▼
#   ReconstructionModel[predicted_class]  →  32³ voxel grid
#
# All 10 reconstruction models are loaded once at startup.
# At inference time only the classifier + one reconstruction model run.
#
# Usage (script)
# --------------
#   python inference.py --input path/to/object.pt
#   python inference.py --input path/to/object.pt --gt_label   # show IoU vs GT
#   python inference.py --cache ./ModelNet10_mv_cache/test --n 5  # random samples
#
# Usage (library)
# ---------------
#   from inference import InferencePipeline
#   pipeline = InferencePipeline()
#   result   = pipeline.run(rgbd_tensor)   # rgbd: (1, 6, 4, H, W)  float32
#   print(result["predicted_class"], result["voxels"].shape)

from __future__ import annotations

import argparse, os, time
from pathlib import Path
from typing import Dict, Optional

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ── Config (must match the training scripts) ────────────────────────────────────
CLS_CKPT_DIR   = "./checkpoints_cls"
RECON_CKPT_DIR = "./checkpoints_recon"
IN_CHANNELS    = 4
N_VIEWS        = 6
CLS_LATENT_DIM = 512
RECON_LATENT   = 1024
NUM_CLASSES    = 10
VOXEL_SIZE     = 32
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ── Architectures (copied verbatim so inference.py is self-contained) ───────────

class _ViewEncoder(nn.Module):
    def _block(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
        )
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels,  32, 2), self._block(32,  64, 2),
            self._block(64,          128, 2), self._block(128, 256, 2),
            self._block(256,         512, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, latent_dim)
        self.act  = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.fc(self.pool(self.features(x)).flatten(1)))


class MultiViewClassifier(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, latent_dim=CLS_LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = _ViewEncoder(in_channels, latent_dim)
        self.head    = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )
    def forward(self, x):
        B, N, C, H, W = x.shape
        feats = self.encoder(x.view(B * N, C, H, W)).view(B, N, -1)
        return self.head(feats.max(dim=1).values)


class _Decoder(nn.Module):
    def _db(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_c), nn.ReLU(inplace=True),
        )
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2 * 2)
        self.network = nn.Sequential(
            self._db(256, 128), self._db(128, 64),
            self._db(64, 32),   self._db(32, 16),
            nn.ConvTranspose3d(16, 1, 1),
        )
    def forward(self, z):
        return self.network(self.fc(z).view(-1, 256, 2, 2, 2))


class _Refiner(nn.Module):
    def _rb(self, ch):
        return nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm3d(ch),
        )
    def __init__(self):
        super().__init__()
        self.conv_in  = nn.Conv3d(1, 32, 3, padding=1)
        self.res1     = self._rb(32); self.res2 = self._rb(32)
        self.conv_out = nn.Conv3d(32, 1, 1); self.relu = nn.ReLU(inplace=True)
    def forward(self, coarse):
        x  = self.relu(self.conv_in(coarse))
        r1 = self.relu(self.res1(x) + x)
        r2 = self.relu(self.res2(r1) + r1)
        return self.conv_out(r2)


class ReconstructionModel(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, latent_dim=RECON_LATENT):
        super().__init__()
        self.encoder = _ViewEncoder(in_channels, latent_dim)
        self.decoder = _Decoder(latent_dim)
        self.refiner = _Refiner()
    def forward(self, x):
        B, N, C, H, W = x.shape
        feats   = self.encoder(x.view(B * N, C, H, W)).view(B, N, -1)
        z       = feats.max(dim=1).values
        coarse  = self.decoder(z)
        refined = self.refiner(coarse)
        return coarse, refined


# ── Pipeline ────────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    Loads all models once and exposes a single `.run()` method.

    Parameters
    ----------
    cls_ckpt_dir   : directory containing ``classifier.pth``
    recon_ckpt_dir : directory containing ``model_<class>.pth`` × 10
    device         : torch device string
    """

    def __init__(
        self,
        cls_ckpt_dir:   str = CLS_CKPT_DIR,
        recon_ckpt_dir: str = RECON_CKPT_DIR,
        device:         str = DEVICE,
    ):
        self.device = device
        self.classes = CLASSES

        # ── Classifier ───────────────────────────────────────────────────────
        cls_path = os.path.join(cls_ckpt_dir, "classifier.pth")
        if not os.path.exists(cls_path):
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {cls_path}\n"
                "Run training_classifier.py first."
            )
        cls_ckpt = torch.load(cls_path, map_location=device, weights_only=True)
        cfg      = cls_ckpt["config"]
        self.classifier = MultiViewClassifier(
            in_channels=cfg["in_channels"],
            latent_dim=cfg["latent_dim"],
            num_classes=cfg["num_classes"],
        ).to(device)
        self.classifier.load_state_dict(cls_ckpt["model_state"])
        self.classifier.eval()
        print(f"Loaded classifier  (val acc={cls_ckpt['val_acc']:.3f}  epoch={cls_ckpt['epoch']})")

        # ── Per-class reconstruction models ───────────────────────────────────
        self.recon_models: Dict[str, ReconstructionModel] = {}
        missing = []
        for cls in CLASSES:
            ckpt_path = os.path.join(recon_ckpt_dir, f"model_{cls}.pth")
            if not os.path.exists(ckpt_path):
                missing.append(cls)
                continue
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
            cfg   = ckpt["config"]
            model = ReconstructionModel(
                in_channels=cfg["in_channels"],
                latent_dim=cfg["latent_dim"],
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self.recon_models[cls] = model
            print(f"Loaded recon model: {cls:<14}  val IoU={ckpt['val_iou']:.4f}  epoch={ckpt['epoch']}")

        if missing:
            print(f"\nWarning: missing reconstruction models for: {missing}")
            print("Run `python training_per_class.py` to train them.\n")

        print(f"\nPipeline ready — {len(self.recon_models)}/10 reconstruction models loaded.")

    @torch.no_grad()
    def run(
        self,
        rgbd: torch.Tensor,
        gt_label: Optional[int] = None,
    ) -> dict:
        """
        Run the full pipeline on one object.

        Parameters
        ----------
        rgbd : (1, 6, 4, H, W)  float32  RGB-D input (all 6 views)
        gt_label : optional ground-truth class index for IoU reporting

        Returns
        -------
        {
            "predicted_class"  : str    class name chosen by the classifier
            "predicted_label"  : int    class index
            "class_probs"      : (10,)  float32 softmax probabilities
            "voxels"           : (32,32,32) bool  reconstructed occupancy
            "voxel_logits"     : (1,1,32,32,32) float32
            "correct"          : bool | None   (only if gt_label given)
            "iou"              : float | None  (only if gt_label given)
            "elapsed_ms"       : float
        }
        """
        rgbd = rgbd.to(self.device)
        t0   = time.time()

        # 1. Classify
        amp_on  = self.device == "cuda"
        with torch.amp.autocast("cuda", enabled=amp_on):
            cls_logits = self.classifier(rgbd)             # (1, 10)

        probs          = torch.softmax(cls_logits, dim=1)[0]   # (10,)
        predicted_label = int(probs.argmax().item())
        predicted_class = CLASSES[predicted_label]

        # 2. Route to the corresponding reconstruction model
        if predicted_class not in self.recon_models:
            raise RuntimeError(
                f"Reconstruction model for '{predicted_class}' is not loaded. "
                "Run training_per_class.py for that class first."
            )

        with torch.amp.autocast("cuda", enabled=amp_on):
            _, refined_logits = self.recon_models[predicted_class](rgbd)

        voxels = (torch.sigmoid(refined_logits[0, 0]) > 0.5).cpu()  # (32,32,32) bool

        elapsed_ms = (time.time() - t0) * 1000

        # 3. Optional: compute IoU against GT voxels if label is provided
        iou     = None
        correct = None
        if gt_label is not None:
            correct = (predicted_label == gt_label)

        result = {
            "predicted_class":  predicted_class,
            "predicted_label":  predicted_label,
            "class_probs":      probs.cpu(),
            "voxels":           voxels,
            "voxel_logits":     refined_logits.cpu(),
            "correct":          correct,
            "iou":              iou,
            "elapsed_ms":       elapsed_ms,
        }
        return result

    def evaluate(self, cache_dir: str, n_samples: Optional[int] = None) -> dict:
        """
        Evaluate the full pipeline on a cache split directory.

        Returns per-class and overall accuracy + IoU.
        """
        from modelnet10_mv_dataset import MultiViewModelNet10Dataset, ToFloat

        ds = MultiViewModelNet10Dataset(cache_dir, transform=ToFloat())
        indices = list(range(len(ds)))
        if n_samples is not None:
            import random; random.shuffle(indices)
            indices = indices[:n_samples]

        per_class_correct = {c: 0 for c in CLASSES}
        per_class_total   = {c: 0 for c in CLASSES}
        per_class_iou     = {c: [] for c in CLASSES}

        for i, idx in enumerate(indices):
            item      = ds[idx]
            gt_label  = int(item["label"])
            gt_class  = CLASSES[gt_label]
            gt_voxels = (item["voxels"][0] > 0.5)   # (32,32,32) bool

            rgb   = item["rgb"].unsqueeze(0)    # (1,6,3,H,W)
            depth = item["depth"].unsqueeze(0)  # (1,6,1,H,W)
            rgbd  = torch.cat([rgb, depth], dim=2)  # (1,6,4,H,W)

            result = self.run(rgbd, gt_label=gt_label)

            per_class_total[gt_class]   += 1
            per_class_correct[gt_class] += int(result["predicted_class"] == gt_class)

            # IoU only when the correct class model was used
            if result["predicted_class"] == gt_class:
                pred = result["voxels"]
                inter = (pred & gt_voxels).sum().item()
                union = (pred | gt_voxels).sum().item()
                per_class_iou[gt_class].append(inter / (union + 1e-8))

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(indices)}] evaluated so far ...")

        # Aggregate
        print(f"\n{'Class':<14} {'Samples':>7}  {'Cls Acc':>8}  {'Recon IoU (correct cls only)':>28}")
        print("-" * 65)
        all_acc, all_iou = [], []
        for cls in CLASSES:
            n   = per_class_total[cls]
            acc = per_class_correct[cls] / max(n, 1)
            iou = float(np.mean(per_class_iou[cls])) if per_class_iou[cls] else float("nan")
            print(f"  {cls:<12}  {n:>7}  {acc*100:>7.1f}%  {iou:>28.4f}")
            if n > 0:
                all_acc.append(acc)
            if per_class_iou[cls]:
                all_iou.extend(per_class_iou[cls])

        overall_acc = float(np.mean(all_acc))
        overall_iou = float(np.mean(all_iou)) if all_iou else float("nan")
        print(f"\n  Overall accuracy : {overall_acc*100:.1f}%")
        print(f"  Overall IoU      : {overall_iou:.4f}  (samples routed to correct model)")

        return {
            "per_class_acc": {c: per_class_correct[c] / max(per_class_total[c],1) for c in CLASSES},
            "per_class_iou": {c: float(np.mean(v)) if v else float("nan") for c, v in per_class_iou.items()},
            "overall_acc":   overall_acc,
            "overall_iou":   overall_iou,
        }


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _visualise(item, result, out_path):
    """Save a quick visualisation of input views + predicted voxels."""
    VIEW_LABELS = ["Front", "Right", "Back", "Left", "Top", "Bottom"]
    rgb     = item["rgb"]       # (6,3,H,W)  float32
    vox     = result["voxels"].numpy()   # (32,32,32)
    R       = vox.shape[0]
    slices  = [R//4, R//2, 3*R//4]

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(
        f"Predicted: {result['predicted_class']}  "
        f"(conf={result['class_probs'].max()*100:.1f}%)  "
        f"| {result['elapsed_ms']:.0f} ms",
        fontsize=13,
    )

    # Row 1: input views
    for v in range(6):
        ax = fig.add_subplot(2, 6, v + 1)
        ax.imshow(rgb[v].permute(1,2,0).numpy())
        ax.set_title(VIEW_LABELS[v], fontsize=8); ax.axis("off")

    # Row 2: voxel slices
    for i, sl in enumerate(slices):
        ax = fig.add_subplot(2, 6, 6 + i + 1)
        ax.imshow(vox[:, :, sl], cmap="gray", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"XY z={sl}", fontsize=8); ax.axis("off")

    # 3D scatter
    ax3d = fig.add_subplot(2, 6, (10, 12), projection="3d")
    occ  = np.argwhere(vox)
    if len(occ):
        ax3d.scatter(occ[:,0], occ[:,1], occ[:,2], c=occ[:,2], cmap="viridis", s=2, alpha=0.5)
    ax3d.set_title("3D voxels", fontsize=8); ax3d.tick_params(labelsize=5)
    ax3d.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved visualisation → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGB-D → voxel inference pipeline")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",  help="Path to a single .pt cache file")
    group.add_argument("--cache",  help="Path to a cache split dir; runs evaluation")
    parser.add_argument("--n",          type=int, default=None, help="Max samples for --cache eval")
    parser.add_argument("--out",        default="inference_output.png")
    parser.add_argument("--cls_dir",    default=CLS_CKPT_DIR)
    parser.add_argument("--recon_dir",  default=RECON_CKPT_DIR)
    args = parser.parse_args()

    pipeline = InferencePipeline(
        cls_ckpt_dir=args.cls_dir,
        recon_ckpt_dir=args.recon_dir,
    )

    if args.input:
        # ── Single file ──────────────────────────────────────────────────────
        from modelnet10_mv_dataset import ToFloat
        item     = torch.load(args.input, map_location="cpu", weights_only=True)
        item     = ToFloat()(item)
        gt_label = int(item["label"])

        rgb   = item["rgb"].unsqueeze(0)
        depth = item["depth"].unsqueeze(0)
        rgbd  = torch.cat([rgb, depth], dim=2)   # (1,6,4,H,W)

        result = pipeline.run(rgbd, gt_label=gt_label)

        print(f"\nGT class       : {CLASSES[gt_label]}")
        print(f"Predicted class: {result['predicted_class']}  "
              f"({'✓' if result['correct'] else '✗'})")
        print(f"Elapsed        : {result['elapsed_ms']:.1f} ms")
        print("\nClass probabilities:")
        for i, (cls, p) in enumerate(zip(CLASSES, result["class_probs"])):
            bar = "█" * int(p * 40)
            print(f"  {cls:<14} {p*100:5.1f}%  {bar}")

        _visualise(item, result, args.out)

    else:
        # ── Dataset evaluation ───────────────────────────────────────────────
        pipeline.evaluate(args.cache, n_samples=args.n)
