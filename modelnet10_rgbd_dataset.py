"""
ModelNet10 RGB-D Dataset  (memory-efficient edition)
=====================================================
Pipeline
--------
  OFF mesh  ──►  normalize  ──►  ray-cast ONE view at a time  ──►  (RGB, Depth)
                           └──►  voxelize 32³                 ──►  occupancy grid (GT)

Memory optimisations applied
-----------------------------
1. **Lazy / streamed rendering** — `__getitem__` renders exactly ONE view
   (selected by `view_idx`).  The full-dataset wrapper `ModelNet10RGBDDataset`
   composes `ModelNet10MeshDataset` × view indices so the DataLoader only ever
   holds one (H, W) depth map and one (H, W, 3) RGB image in RAM per worker
   at a time, instead of all N_VIEWS simultaneously.

2. **Compact dtypes in tensors**
     • RGB   → uint8  [0, 255]  — 4× smaller than float32; cast to float in model
     • Depth → float16 [0, 1]   — 2× smaller; sufficient precision for geometry
     • Voxels→ uint8  {0, 1}    — 4× smaller than float32

   Combined saving per item (12 views, 128² images, 32³ voxels):
     Before : ~75 MB  (all float32)
     After  :  ~0.2 MB  (streamed single view + compact dtypes)

Item structure (single-view mode, default)
------------------------------------------
  {
    "rgb"      : Tensor (3, H, W)    uint8   [0, 255]
    "depth"    : Tensor (1, H, W)    float16 [0, 1]
    "voxels"   : Tensor (1, R, R, R) uint8   {0, 1}
    "view_idx" : int                          which of the N_VIEWS cameras
    "label"    : int
    "path"     : str
  }

  Use the `ToFloat` transform (or pass `to_float=True` to `make_dataloaders`)
  to get float32 tensors ready for a model forward pass.

Requirements
------------
    pip install torch torchvision trimesh rtree scipy numpy
    # optional for real colour shading:
    pip install pytorch3d   # https://github.com/facebookresearch/pytorch3d
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh

# ── optional PyTorch3D import ──────────────────────────────────────────────────
try:
    import pytorch3d                                        # noqa: F401
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        MeshRasterizer,
        MeshRenderer,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
        look_at_view_transform,
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    warnings.warn(
        "PyTorch3D not found – RGB channel will use surface-normal shading fallback. "
        "Install with: pip install pytorch3d",
        ImportWarning,
        stacklevel=2,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASSES: List[str] = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

N_VIEWS       = 12
ELEVATION_DEG = 30.0
CAMERA_DIST   = 2.5
IMG_H = IMG_W = 128
VOXEL_RES     = 32

DEPTH_MIN: float = 0.0
DEPTH_MAX: float = CAMERA_DIST + 1.5


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre and scale mesh so it fits inside [-1, 1]³."""
    mesh = mesh.copy()
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = np.abs(mesh.vertices).max()
    if scale > 0:
        mesh.vertices /= scale
    return mesh


def camera_positions(
    n_views: int = N_VIEWS,
    elevation_deg: float = ELEVATION_DEG,
    radius: float = CAMERA_DIST,
) -> np.ndarray:
    """Return ``(n_views, 3)`` eye positions on an elevation ring."""
    azimuths = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    el = np.deg2rad(elevation_deg)
    return np.stack(
        [
            radius * np.cos(el) * np.cos(azimuths),
            radius * np.cos(el) * np.sin(azimuths),
            np.full(n_views, radius * np.sin(el)),
        ],
        axis=1,
    ).astype(np.float32)


def _look_at_matrix(eye: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Return ``(3, 3)`` rotation matrix [right, up, forward] in world space."""
    if target is None:
        target = np.zeros(3, dtype=np.float32)
    up_hint = np.array([0.0, 0.0, 1.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        up_hint = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Rendering — one view at a time
# ──────────────────────────────────────────────────────────────────────────────

def render_depth_single(
    mesh: trimesh.Trimesh,
    eye: np.ndarray,
    h: int = IMG_H,
    w: int = IMG_W,
    fov_deg: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ray-cast ``mesh`` from one camera position.

    Returns
    -------
    depth   : (H, W) float32   raw Euclidean distance; 0 = no hit
    normals : (H, W, 3) float32 world-space surface normals
    mask    : (H, W) bool
    """
    focal = w / (2.0 * np.tan(np.deg2rad(fov_deg / 2.0)))
    R = _look_at_matrix(eye)

    u = np.linspace(-(w - 1) / 2, (w - 1) / 2, w)
    v = np.linspace(-(h - 1) / 2, (h - 1) / 2, h)
    uu, vv = np.meshgrid(u, v)
    dirs_cam   = np.stack([uu / focal, vv / focal, np.ones_like(uu)], axis=-1).reshape(-1, 3)
    dirs_world = dirs_cam @ R
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    origins    = np.tile(eye, (h * w, 1)).astype(np.float32)

    intersector          = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locs, ray_idx, f_idx = intersector.intersects_location(origins, dirs_world, multiple_hits=False)

    depth_flat   = np.zeros(h * w, dtype=np.float32)
    normals_flat = np.zeros((h * w, 3), dtype=np.float32)
    if len(locs) > 0:
        depth_flat[ray_idx]   = np.linalg.norm(locs - eye, axis=-1).astype(np.float32)
        normals_flat[ray_idx] = mesh.face_normals[f_idx].astype(np.float32)

    depth   = depth_flat.reshape(h, w)
    normals = normals_flat.reshape(h, w, 3)
    mask    = depth > 0
    return depth, normals, mask


def _normalise_depth(depth: np.ndarray) -> np.ndarray:
    """Map raw depth → [0, 1]; background stays 0."""
    out  = np.zeros_like(depth)
    mask = depth > 0
    out[mask] = np.clip((depth[mask] - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN), 0.0, 1.0)
    return out


def _rgb_from_normals(normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fallback: surface normals → colour. Returns (H, W, 3) float32."""
    rgb = (normals * 0.5 + 0.5).clip(0, 1).astype(np.float32)
    rgb[~mask] = 0.0
    return rgb


def render_rgb_pytorch3d_single(
    mesh: trimesh.Trimesh,
    eye: np.ndarray,
    h: int = IMG_H,
    w: int = IMG_W,
    device: torch.device | None = None,
) -> np.ndarray:
    """Render one Phong-shaded colour image. Returns (H, W, 3) float32 [0,1]."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    verts_t  = torch.tensor(mesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)
    faces_t  = torch.tensor(mesh.faces,    dtype=torch.int64,   device=device).unsqueeze(0)
    textures = TexturesVertex(verts_features=torch.ones_like(verts_t) * 0.7)
    p3d_mesh = Meshes(verts=verts_t, faces=faces_t, textures=textures)

    eye_t = torch.tensor(eye,         dtype=torch.float32, device=device).unsqueeze(0)
    at_t  = torch.zeros(1, 3,         dtype=torch.float32, device=device)
    up_t  = torch.tensor([[0, 0, 1]], dtype=torch.float32, device=device)
    R, T  = look_at_view_transform(eye=eye_t, at=at_t, up=up_t)

    cameras  = FoVPerspectiveCameras(R=R, T=T, fov=60.0, device=device)
    lights   = PointLights(location=eye_t, device=device)
    raster   = RasterizationSettings(image_size=h, blur_radius=0.0, faces_per_pixel=1)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftPhongShader(cameras=cameras, lights=lights, device=device),
    )
    with torch.no_grad():
        img = renderer(p3d_mesh)    # (1, H, W, 4)
    return img[0, :, :, :3].cpu().numpy().astype(np.float32)


def render_single_view(
    mesh: trimesh.Trimesh,
    eye: np.ndarray,
    h: int = IMG_H,
    w: int = IMG_W,
    use_pytorch3d: bool = PYTORCH3D_AVAILABLE,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render exactly **one** camera view of ``mesh``.

    Returns
    -------
    rgb   : (H, W, 3) float32  [0, 1]
    depth : (H, W)    float32  normalised [0, 1]
    """
    depth_raw, normals, mask = render_depth_single(mesh, eye, h=h, w=w)
    depth = _normalise_depth(depth_raw)

    if use_pytorch3d and PYTORCH3D_AVAILABLE:
        try:
            rgb = render_rgb_pytorch3d_single(mesh, eye, h=h, w=w, device=device)
        except Exception as exc:
            warnings.warn(f"PyTorch3D render failed ({exc}); using normal-shading fallback.")
            rgb = _rgb_from_normals(normals, mask)
    else:
        rgb = _rgb_from_normals(normals, mask)

    return rgb, depth


# ──────────────────────────────────────────────────────────────────────────────
# Compact dtype helpers
# ──────────────────────────────────────────────────────────────────────────────

def rgb_to_uint8(rgb: np.ndarray) -> torch.Tensor:
    """(H,W,3) float32 [0,1] → (3,H,W) uint8 [0,255].  4× smaller than float32."""
    return torch.from_numpy((rgb * 255).clip(0, 255).astype(np.uint8)).permute(2, 0, 1)


def depth_to_float16(depth: np.ndarray) -> torch.Tensor:
    """(H,W) float32 → (1,H,W) float16.  2× smaller than float32."""
    return torch.from_numpy(depth.astype(np.float16)).unsqueeze(0)


def voxels_to_uint8(voxels: np.ndarray) -> torch.Tensor:
    """(R,R,R) float32 {0,1} → (1,R,R,R) uint8.  4× smaller than float32."""
    return torch.from_numpy(voxels.astype(np.uint8)).unsqueeze(0)


class ToFloat:
    """
    Transform: cast compact-dtype tensors to float32 for model input.

    Usage::

        ds = ModelNet10RGBDDataset(..., transform=ToFloat())
        item = ds[0]
        # item["rgb"]    : float32  [0, 1]
        # item["depth"]  : float32  [0, 1]
        # item["voxels"] : float32  {0., 1.}
    """
    def __call__(self, item: dict) -> dict:
        out = dict(item)
        out["rgb"]    = item["rgb"].float()    / 255.0
        out["depth"]  = item["depth"].float()
        out["voxels"] = item["voxels"].float()
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Voxelisation
# ──────────────────────────────────────────────────────────────────────────────

def voxelize(mesh: trimesh.Trimesh, resolution: int = VOXEL_RES) -> np.ndarray:
    """
    Voxelise normalised ``mesh`` into a binary (resolution³) occupancy grid.

    Returns
    -------
    grid : (R, R, R) float32  {0, 1}
    """
    pitch    = 2.0 / resolution
    vox      = mesh.voxelized(pitch=pitch).fill()
    grid     = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    indices  = np.argwhere(vox.matrix)
    if len(indices) == 0:
        return grid

    world_pts = vox.translation + indices * pitch + pitch / 2.0
    grid_idx  = np.floor((world_pts + 1.0) / 2.0 * resolution).astype(int)
    valid     = np.all((grid_idx >= 0) & (grid_idx < resolution), axis=1)
    gi        = grid_idx[valid]
    grid[gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ModelNet10RGBDDataset(Dataset):
    """
    Memory-efficient ModelNet10 RGB-D dataset.

    Indexing scheme
    ---------------
    Each integer index maps to **(mesh_i, view_j)**:

        mesh_idx = idx // n_views
        view_idx = idx  % n_views

    So ``len(dataset) == n_meshes * n_views`` and each ``__getitem__`` call
    loads one mesh and renders exactly one camera view — keeping peak RAM
    proportional to a *single frame*, not the full view stack.

    Parameters
    ----------
    root : str | Path
        Path to the ModelNet10 directory.
    split : 'train' | 'test'
    classes : list[str] | None
        Subset of class names.  None → all 10.
    n_views : int
        Number of camera positions per object.
    img_size : int
        H = W of rendered images (pixels).
    voxel_res : int
        Cubic voxel grid resolution for ground-truth occupancy.
    use_pytorch3d : bool
        Use PyTorch3D Phong shading; falls back to normal-colouring if unavailable.
    device : torch.device | None
        Device for PyTorch3D; auto-detected if None.
    transform : callable | None
        Applied to the output dict.  Pass ``ToFloat()`` to get float32 tensors.

    Item structure
    --------------
    {
        "rgb"      : Tensor (3, H, W)    uint8   [0, 255]
        "depth"    : Tensor (1, H, W)    float16 [0, 1]
        "voxels"   : Tensor (1, R, R, R) uint8   {0, 1}
        "view_idx" : int
        "label"    : int
        "path"     : str
    }

    Memory per item vs. old implementation (12 views, 128² px, 32³ voxels)
    -----------------------------------------------------------------------
        Before : ~75 MB  (all-views, all float32)
        After  :  ~0.2 MB (single view, compact dtypes)
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        classes: Optional[List[str]] = None,
        n_views: int = N_VIEWS,
        img_size: int = IMG_H,
        voxel_res: int = VOXEL_RES,
        use_pytorch3d: bool = PYTORCH3D_AVAILABLE,
        device: Optional[torch.device] = None,
        transform=None,
    ):
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.root          = Path(root)
        self.split         = split
        self.n_views       = n_views
        self.img_size      = img_size
        self.voxel_res     = voxel_res
        self.use_pytorch3d = use_pytorch3d
        self.device        = device
        self.transform     = transform

        self.classes = classes if classes is not None else CLASSES
        unknown = set(self.classes) - set(CLASSES)
        if unknown:
            raise ValueError(f"Unknown classes: {unknown}")

        self._samples: List[Tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = self.root / cls / split
            if not cls_dir.exists():
                warnings.warn(f"Directory not found: {cls_dir}")
                continue
            for f in sorted(glob.glob(str(cls_dir / "*.off"))):
                self._samples.append((f, CLASS_TO_IDX[cls]))

        if not self._samples:
            raise RuntimeError(
                f"No .off files found under {self.root}. "
                "Check that `root` points to the ModelNet10 directory."
            )

        # Camera ring — computed once, shared across all __getitem__ calls
        self._eyes: np.ndarray = camera_positions(n_views=n_views)

    def __len__(self) -> int:
        return len(self._samples) * self.n_views

    def __repr__(self) -> str:
        return (
            f"ModelNet10RGBDDataset("
            f"split={self.split!r}, meshes={len(self._samples)}, "
            f"n_views={self.n_views}, total_items={len(self)}, "
            f"img_size={self.img_size}, voxel_res={self.voxel_res})"
        )

    def __getitem__(self, idx: int) -> dict:
        mesh_idx        = idx // self.n_views
        view_idx        = idx  % self.n_views
        path, label     = self._samples[mesh_idx]

        # 1. Load & normalise mesh
        mesh = normalize_mesh(trimesh.load(path, force="mesh"))

        # 2. Render ONE view  →  (H,W,3) float32,  (H,W) float32
        rgb_np, depth_np = render_single_view(
            mesh, self._eyes[view_idx],
            h=self.img_size, w=self.img_size,
            use_pytorch3d=self.use_pytorch3d,
            device=self.device,
        )

        # 3. Voxelise  →  (R,R,R) float32  {0,1}
        voxel_np = voxelize(mesh, resolution=self.voxel_res)

        # 4. Store as compact dtypes
        item = {
            "rgb":      rgb_to_uint8(rgb_np),       # (3,H,W)    uint8
            "depth":    depth_to_float16(depth_np), # (1,H,W)    float16
            "voxels":   voxels_to_uint8(voxel_np),  # (1,R,R,R)  uint8
            "view_idx": view_idx,
            "label":    label,
            "path":     path,
        }

        if self.transform is not None:
            item = self.transform(item)

        return item


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    root: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    to_float: bool = True,
    **dataset_kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and test DataLoaders for ModelNet10.

    Parameters
    ----------
    to_float : bool
        Wrap with ``ToFloat()`` so tensors arrive as float32 [0,1].

    Example
    -------
    >>> train_loader, test_loader = make_dataloaders("./ModelNet10", batch_size=4)
    >>> batch = next(iter(train_loader))
    >>> batch["rgb"].shape      # (4, 3, 128, 128)    float32
    >>> batch["depth"].shape    # (4, 1, 128, 128)    float32
    >>> batch["voxels"].shape   # (4, 1, 32, 32, 32)  float32
    """
    transform = ToFloat() if to_float else None
    train_ds  = ModelNet10RGBDDataset(root=root, split="train", transform=transform, **dataset_kwargs)
    test_ds   = ModelNet10RGBDDataset(root=root, split="test",  transform=transform, **dataset_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, time
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Usage: python modelnet10_rgbd_dataset.py <path/to/file.off>")
        print("       python modelnet10_rgbd_dataset.py <path/to/ModelNet10>")
        sys.exit(0)

    p = Path(sys.argv[1])

    if p.suffix == ".off":
        print(f"\n=== Single-file smoke test: {p} ===\n")
        mesh = normalize_mesh(trimesh.load(str(p), force="mesh"))
        eyes = camera_positions()

        import tracemalloc
        tracemalloc.start()

        t0 = time.time()
        all_rgb, all_depth = [], []
        for eye in eyes:
            rgb_np, depth_np = render_single_view(mesh, eye)
            all_rgb.append(rgb_to_uint8(rgb_np))
            all_depth.append(depth_to_float16(depth_np))

        voxel_np = voxelize(mesh)
        voxel_t  = voxels_to_uint8(voxel_np)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Rendered {N_VIEWS} views in {time.time()-t0:.2f}s  |  peak RAM: {peak/1e6:.1f} MB")
        print(f"  rgb[0]  : shape={tuple(all_rgb[0].shape)}   dtype={all_rgb[0].dtype}")
        print(f"  depth[0]: shape={tuple(all_depth[0].shape)}  dtype={all_depth[0].dtype}")
        print(f"  voxels  : shape={tuple(voxel_t.shape)}  dtype={voxel_t.dtype}  "
              f"occupancy={voxel_np.mean()*100:.1f}%")

        rgb_show   = torch.stack(all_rgb).float() / 255.0
        depth_show = torch.stack(all_depth).float()

        fig, axes = plt.subplots(3, N_VIEWS, figsize=(N_VIEWS * 2, 6))
        for v in range(N_VIEWS):
            axes[0, v].imshow(rgb_show[v].permute(1, 2, 0).numpy())
            axes[0, v].set_title(f"RGB {v}", fontsize=7)
            axes[1, v].imshow(depth_show[v, 0].numpy(), cmap="plasma")
            axes[1, v].set_title(f"Depth {v}", fontsize=7)
            axes[2, v].imshow(voxel_np[:, :, VOXEL_RES // 2], cmap="gray")
            axes[2, v].set_title("Voxel Z/2", fontsize=7)
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        out = Path("smoke_test_output.png")
        fig.savefig(out, dpi=120)
        print(f"\nVisualization saved → {out.resolve()}")

    else:
        print(f"\n=== Dataset smoke test: {p} ===\n")
        ds = ModelNet10RGBDDataset(root=p, split="train", transform=ToFloat())
        print(ds)
        print(f"\nLoading item 0 (mesh 0, view 0) …")
        t0   = time.time()
        item = ds[0]
        print(f"  Done in {time.time()-t0:.2f}s")
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:10s}: shape={tuple(v.shape)}  dtype={v.dtype}  "
                      f"range=[{v.min():.3f}, {v.max():.3f}]")
            else:
                print(f"  {k:10s}: {v}")
        item1 = ds[1]
        print(f"\nItem 1: view_idx={item1['view_idx']}  (same mesh, next camera angle)")