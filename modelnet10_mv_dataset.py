"""
modelnet10_mv_dataset.py
========================
Dataset wrapper for the multi-view .pt cache produced by preprocess_multiview.py.

Each .pt file (one per object) contains:
    {
        "rgb"    : (6, 3, H, W)    uint8   [0, 255]
        "depth"  : (6, 1, H, W)    float16 [0, 1]
        "voxels" : (1, R, R, R)    uint8   {0, 1}
        "label"  : long tensor
        "path"   : str
    }

After ToFloat():
    rgb    → float32 [0, 1]
    depth  → float32 [0, 1]
    voxels → float32 {0., 1.}

DataLoader batch shapes (batch_size=B):
    rgb    : (B, 6, 3, H, W)
    depth  : (B, 6, 1, H, W)
    voxels : (B, 1, R, R, R)
    label  : (B,)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset


class MultiViewModelNet10Dataset(Dataset):
    """
    Read pre-rendered multi-view items from a cache directory.

    Parameters
    ----------
    cache_dir : str | Path
        A split directory produced by preprocess_multiview.py,
        e.g. ``./ModelNet10_mv_cache/train``.
        Expected structure::

            cache_dir/
              bathtub/  bathtub_0001.pt  bathtub_0002.pt  …
              bed/      bed_0001.pt      …
              …

    transform : callable | None
        Applied to each item dict after loading.
        Pass ``ToFloat()`` to get float32 tensors ready for a model.
    label_filter : list[int] | None
        If given, only items whose ``label`` is in this set are returned.
        Useful for single-class debugging.
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
                "Run `python preprocess_multiview.py` first."
            )

        # Collect all .pt files (may be nested under class subdirectories)
        all_files = sorted(self.cache_dir.rglob("*.pt"))
        if not all_files:
            raise RuntimeError(
                f"No .pt files found under {self.cache_dir}.\n"
                "Run `python preprocess_multiview.py` first."
            )

        if self.label_filter is not None:
            self._files: List[Path] = []
            for f in all_files:
                item = torch.load(f, map_location="cpu", weights_only=True)
                if int(item["label"]) in self.label_filter:
                    self._files.append(f)
        else:
            self._files = all_files

        print(
            f"MultiViewModelNet10Dataset | {self.cache_dir.name} "
            f"| {len(self._files):,} objects × 6 views"
        )

    def __len__(self) -> int:
        return len(self._files)

    def __repr__(self) -> str:
        return (
            f"MultiViewModelNet10Dataset("
            f"cache={self.cache_dir}, objects={len(self)}, "
            f"transform={self.transform})"
        )

    def __getitem__(self, idx: int) -> dict:
        item = torch.load(self._files[idx], map_location="cpu", weights_only=True)
        item["label"] = int(item["label"])
        if self.transform is not None:
            item = self.transform(item)
        return item


# ──────────────────────────────────────────────────────────────────────────────
# ToFloat transform
# ──────────────────────────────────────────────────────────────────────────────

class ToFloat:
    """
    Cast compact-dtype tensors to float32 for model input.

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


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def make_mv_dataloaders(
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
    Build train + test DataLoaders from the multi-view cache.

    Parameters
    ----------
    cache_root : directory containing ``train/`` and ``test/`` subdirectories.

    Example
    -------
    >>> train_loader, test_loader = make_mv_dataloaders("./ModelNet10_mv_cache")
    >>> batch = next(iter(train_loader))
    >>> batch["rgb"].shape      # (8, 6, 3, 128, 128)   float32
    >>> batch["depth"].shape    # (8, 6, 1, 128, 128)   float32
    >>> batch["voxels"].shape   # (8, 1,  32,  32,  32) float32
    """
    cache_root = Path(cache_root)
    transform  = ToFloat() if to_float else None

    train_ds = MultiViewModelNet10Dataset(cache_root / "train", transform=transform, **dataset_kwargs)
    test_ds  = MultiViewModelNet10Dataset(cache_root / "test",  transform=transform, **dataset_kwargs)

    use_pin = pin_memory and torch.cuda.is_available()

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    if num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **loader_kwargs)
    return train_loader, test_loader
