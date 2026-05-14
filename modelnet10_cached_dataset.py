"""
modelnet10_cached_dataset.py
============================
Thin Dataset wrapper around pre-rendered .pt cache files.
Run `preprocess_dataset.py` once before using this.

Each .pt file contains:
    {
        "rgb":      (3, H, W)    uint8   [0, 255]
        "depth":    (1, H, W)    float16 [0, 1]
        "voxels":   (1, R, R, R) uint8   {0, 1}
        "view_idx": int scalar tensor
        "label":    int scalar tensor
        "path":     str
    }

Speed vs. on-the-fly rendering
--------------------------------
On-the-fly : ~200–2000 ms / item   (mesh load + ray-cast + voxelize)
Cached     :    ~1–5   ms / item   (torch.load of a small .pt)
→ GPU utilisation goes from ~5 % to >90 % on a typical workstation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset


class CachedModelNet10Dataset(Dataset):
    """
    Read pre-rendered items from a directory of .pt files.

    Parameters
    ----------
    cache_dir : str | Path
        Directory produced by preprocess_dataset.py, e.g. ``./ModelNet10_cache/train``.
    transform : callable | None
        Applied to each item dict.  Use ``ToFloat`` from the original dataset
        module to cast uint8/float16 tensors to float32 [0,1].
    label_filter : list[int] | None
        Only return items whose label is in this list (subset of classes).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        transform: Optional[Callable] = None,
        label_filter: Optional[List[int]] = None,
    ):
        self.cache_dir    = Path(cache_dir)
        self.transform    = transform
        self.label_filter = set(label_filter) if label_filter else None

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_dir}\n"
                "Run `python preprocess_dataset.py` first."
            )

        all_files = sorted(self.cache_dir.glob("*.pt"))
        if not all_files:
            raise RuntimeError(
                f"No .pt files found in {self.cache_dir}.\n"
                "Run `python preprocess_dataset.py` first."
            )

        if self.label_filter is not None:
            # Fast filter: peek at the label in each file's filename is not
            # reliable, so we build the filtered list by scanning labels.
            # For large datasets load just the label tensor (cheap: it's tiny).
            self._files: List[Path] = []
            for f in all_files:
                item = torch.load(f, map_location="cpu", weights_only=True)
                if int(item["label"]) in self.label_filter:
                    self._files.append(f)
        else:
            self._files = all_files

        print(f"CachedModelNet10Dataset | {self.cache_dir.name} | {len(self._files):,} items")

    def __len__(self) -> int:
        return len(self._files)

    def __repr__(self) -> str:
        return (
            f"CachedModelNet10Dataset("
            f"cache={self.cache_dir}, "
            f"items={len(self)}, "
            f"transform={self.transform})"
        )

    def __getitem__(self, idx: int) -> dict:
        item = torch.load(self._files[idx], map_location="cpu", weights_only=True)
        # Convert scalar tensors to plain Python ints for DataLoader collation
        item["view_idx"] = int(item["view_idx"])
        item["label"]    = int(item["label"])
        if self.transform is not None:
            item = self.transform(item)
        return item


# ─────────────────────────────────────────────────────────────────────────────
# ToFloat transform (same API as the original dataset module)
# ─────────────────────────────────────────────────────────────────────────────

class ToFloat:
    """
    Cast compact-dtype tensors to float32 ready for a model forward pass.

        rgb    : uint8  [0,255]  →  float32 [0,1]
        depth  : float16 [0,1]  →  float32 [0,1]
        voxels : uint8  {0,1}   →  float32 {0.,1.}
    """
    def __call__(self, item: dict) -> dict:
        out = dict(item)
        out["rgb"]    = item["rgb"].float() / 255.0
        out["depth"]  = item["depth"].float()
        out["voxels"] = item["voxels"].float()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def make_cached_dataloaders(
    cache_root: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    to_float: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    **dataset_kwargs,
):
    """
    Build train + test DataLoaders from a pre-rendered cache.

    Parameters
    ----------
    cache_root : path that contains ``train/`` and ``test/`` subdirectories.
    prefetch_factor : how many batches each worker pre-loads ahead of time.
                      Higher = more RAM, less GPU stall.  4–8 is a good range.
    persistent_workers : keep worker processes alive between epochs (avoids
                         re-spawning overhead; requires num_workers > 0).

    Example
    -------
    >>> train_loader, test_loader = make_cached_dataloaders("./ModelNet10_cache")
    >>> batch = next(iter(train_loader))
    >>> batch["rgb"].shape      # (8, 3, 128, 128)   float32  [0,1]
    >>> batch["depth"].shape    # (8, 1, 128, 128)   float32  [0,1]
    >>> batch["voxels"].shape   # (8, 1, 32, 32, 32) float32  {0.,1.}
    """
    cache_root = Path(cache_root)
    transform  = ToFloat() if to_float else None

    train_ds = CachedModelNet10Dataset(cache_root / "train", transform=transform, **dataset_kwargs)
    test_ds  = CachedModelNet10Dataset(cache_root / "test",  transform=transform, **dataset_kwargs)

    common_pin_memory = pin_memory and torch.cuda.is_available()

    if num_workers > 0:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=common_pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=common_pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=common_pin_memory,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=common_pin_memory,
        )
    return train_loader, test_loader
