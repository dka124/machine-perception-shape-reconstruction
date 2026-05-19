"""
preprocess_multiview.py
=======================
Renders 6 fixed views of every ModelNet10 mesh and saves one .pt file
per object.  Run this ONCE before training.

Random horizontal rotation
--------------------------
Each object is rotated by a uniformly random angle in [0°, 360°) around the
vertical (Z) axis before rendering and voxelisation.  Because BOTH the RGB-D
views and the voxel grid are derived from the same rotated mesh, they remain
perfectly consistent with each other.  The rotation angle is stored in the
.pt file for reference but is not used during training.

This gives the model exposure to every horizontal orientation of every object
without increasing the dataset size.

Views (applied after rotation)
-------------------------------
  0 — side   azimuth=  0°  elevation=  0° (front)
  1 — side   azimuth= 90°  elevation=  0° (right)
  2 — side   azimuth=180°  elevation=  0° (back)
  3 — side   azimuth=270°  elevation=  0° (left)
  4 — top    azimuth=  0°  elevation=+89° (directly above)
  5 — bottom azimuth=  0°  elevation=-89° (directly below)

Each .pt file contains:
    {
        "rgb"          : (6, 3, H, W)    uint8   [0, 255]
        "depth"        : (6, 1, H, W)    float16 [0, 1]
        "voxels"       : (1, R, R, R)    uint8   {0, 1}
        "label"        : int tensor
        "rotation_deg" : float tensor    the random Z-rotation applied (degrees)
        "path"         : str             original .off path
    }

Usage
-----
    python preprocess_multiview.py \
        --root   ./ModelNet10 \
        --cache  ./ModelNet10_mv_cache \
        --workers 8

    # subset -- only train split, only chair + table
    python preprocess_multiview.py \
        --root ./ModelNet10 --cache ./ModelNet10_mv_cache \
        --splits train --classes chair table
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

# ──────────────────────────────────────────────────────────────────────────────
# Constants shared with dataset / training
# ──────────────────────────────────────────────────────────────────────────────

CLASSES: List[str] = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

IMG_H = IMG_W = 128
VOXEL_RES     = 32
CAMERA_DIST   = 2.5
DEPTH_MIN     = 0.0
DEPTH_MAX     = CAMERA_DIST + 1.5

# 6 cameras: 4 sides (elevation=0) + top + bottom
# Elevation clamped to ±89° to avoid the look-at pole singularity
_VIEWS: List[Tuple[float, float]] = [
    (  0.0,  0.0),   # front
    ( 90.0,  0.0),   # right
    (180.0,  0.0),   # back
    (270.0,  0.0),   # left
    (  0.0, 89.0),   # top
    (  0.0,-89.0),   # bottom
]
N_VIEWS = len(_VIEWS)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────

def _camera_eye(azimuth_deg: float, elevation_deg: float, radius: float = CAMERA_DIST) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([
        radius * np.cos(el) * np.cos(az),
        radius * np.cos(el) * np.sin(az),
        radius * np.sin(el),
    ], dtype=np.float32)


def _look_at(eye: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Return (3,3) rotation matrix rows=[right, up, forward] in world space."""
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


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre and scale mesh to fit in [-1, 1]^3."""
    mesh = mesh.copy()
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = np.abs(mesh.vertices).max()
    if scale > 0:
        mesh.vertices /= scale
    return mesh


def rotate_mesh_z(mesh: trimesh.Trimesh, angle_deg: float) -> trimesh.Trimesh:
    """
    Rotate mesh around the vertical (Z) axis by angle_deg degrees.

    Applied AFTER normalisation so the mesh stays centred in [-1, 1]^3.
    Both the rendered views and the voxel grid are derived from the rotated
    mesh, so they remain consistent with each other.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # 3x3 rotation matrix around Z
    Rz = np.array([
        [ cos_a, -sin_a, 0.0],
        [ sin_a,  cos_a, 0.0],
        [  0.0,    0.0,  1.0],
    ], dtype=np.float64)

    mesh = mesh.copy()
    mesh.vertices = (Rz @ mesh.vertices.T).T   # (N,3)
    # Invalidate cached face normals so trimesh recomputes from rotated vertices
    mesh.face_normals   # access once to force recompute
    return mesh


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────

def _render_one_view(
    mesh: trimesh.Trimesh,
    eye: np.ndarray,
    h: int = IMG_H,
    w: int = IMG_W,
    fov_deg: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ray-cast mesh from eye.

    Returns
    -------
    depth   : (H, W) float32  raw distance; 0 = background
    normals : (H, W, 3) float32
    mask    : (H, W) bool
    """
    focal = w / (2.0 * np.tan(np.deg2rad(fov_deg / 2.0)))
    R = _look_at(eye)

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
    out  = np.zeros_like(depth)
    mask = depth > 0
    out[mask] = np.clip((depth[mask] - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN), 0.0, 1.0)
    return out


def _rgb_from_normals(normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Surface-normal shading. Returns (H, W, 3) float32 [0,1]."""
    rgb = (normals * 0.5 + 0.5).clip(0, 1).astype(np.float32)
    rgb[~mask] = 0.0
    return rgb


def render_all_views(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render all 6 camera positions.

    Returns
    -------
    rgb   : (6, H, W, 3) float32  [0, 1]
    depth : (6, H, W)    float32  normalised [0, 1]
    """
    rgb_list, depth_list = [], []
    for az, el in _VIEWS:
        eye              = _camera_eye(az, el)
        d, normals, mask = _render_one_view(mesh, eye)
        depth_list.append(_normalise_depth(d))
        rgb_list.append(_rgb_from_normals(normals, mask))
    return np.stack(rgb_list), np.stack(depth_list)   # (6,H,W,3), (6,H,W)


# ──────────────────────────────────────────────────────────────────────────────
# Voxelisation
# ──────────────────────────────────────────────────────────────────────────────

def voxelize(mesh: trimesh.Trimesh, resolution: int = VOXEL_RES) -> np.ndarray:
    """Returns (R, R, R) float32 {0, 1}."""
    pitch    = 2.0 / resolution
    vox      = mesh.voxelized(pitch=pitch).fill()
    grid     = np.zeros((resolution,) * 3, dtype=np.float32)
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
# Per-file worker
# ──────────────────────────────────────────────────────────────────────────────

def _process_one(args: Tuple[str, int, Path]) -> Optional[str]:
    """
    Load one .off file, apply a random Z-rotation, render 6 views,
    voxelise, and save a .pt file.

    The rotation angle is drawn fresh for every object (each worker has its
    own RNG state seeded from the OS, so parallel workers don't produce
    identical angles).

    Returns the output path on success, or None on failure.
    """
    off_path, label, out_path = args

    if out_path.exists():
        return str(out_path)   # resume-safe: skip already-written files

    try:
        # 1. Load and normalise
        mesh = normalize_mesh(trimesh.load(off_path, force="mesh"))

        # 2. Random horizontal rotation — drawn uniformly from [0°, 360°)
        #    Using numpy's default_rng (not the legacy global RNG) so each
        #    worker process gets an independent stream.
        angle_deg = np.random.default_rng().uniform(0.0, 360.0)
        mesh      = rotate_mesh_z(mesh, angle_deg)

        # 3. Render all 6 views from the rotated mesh
        rgb_np, depth_np = render_all_views(mesh)   # (6,H,W,3), (6,H,W)

        # 4. Voxelise the rotated mesh  →  GT is consistent with the views
        voxel_np = voxelize(mesh)                   # (R,R,R)

        # 5. Convert to compact storage dtypes
        rgb_t   = torch.from_numpy(
            (rgb_np * 255).clip(0, 255).astype(np.uint8)
        ).permute(0, 3, 1, 2)                        # (6,3,H,W)  uint8

        depth_t = torch.from_numpy(
            depth_np.astype(np.float16)
        ).unsqueeze(1)                               # (6,1,H,W)  float16

        voxel_t = torch.from_numpy(
            voxel_np.astype(np.uint8)
        ).unsqueeze(0)                               # (1,R,R,R)  uint8

        item = {
            "rgb":          rgb_t,
            "depth":        depth_t,
            "voxels":       voxel_t,
            "label":        torch.tensor(label, dtype=torch.long),
            "rotation_deg": torch.tensor(angle_deg, dtype=torch.float32),
            "path":         off_path,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(item, out_path)
        return str(out_path)

    except Exception as exc:
        warnings.warn(f"Failed to process {off_path}: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def build_task_list(
    modelnet_root: Path,
    cache_root: Path,
    splits: List[str],
    classes: List[str],
) -> List[Tuple[str, int, Path]]:
    tasks = []
    for split in splits:
        for cls in classes:
            cls_dir = modelnet_root / cls / split
            if not cls_dir.exists():
                warnings.warn(f"Directory not found, skipping: {cls_dir}")
                continue
            label   = CLASS_TO_IDX[cls]
            out_dir = cache_root / split / cls
            for off in sorted(glob.glob(str(cls_dir / "*.off"))):
                stem     = Path(off).stem
                out_path = out_dir / f"{stem}.pt"
                tasks.append((off, label, out_path))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ModelNet10 → multi-view .pt cache with random Z-rotation"
    )
    parser.add_argument("--root",    required=True, help="Path to ModelNet10 directory")
    parser.add_argument("--cache",   required=True, help="Output cache directory")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                        help="Number of parallel worker processes")
    parser.add_argument("--splits",  nargs="+", default=["train", "test"],
                        choices=["train", "test"])
    parser.add_argument("--classes", nargs="+", default=CLASSES, choices=CLASSES)
    args = parser.parse_args()

    modelnet_root = Path(args.root)
    cache_root    = Path(args.cache)

    tasks = build_task_list(modelnet_root, cache_root, args.splits, args.classes)
    print(f"Tasks  : {len(tasks):,}  ({len(args.splits)} splits × {len(args.classes)} classes)")
    print(f"Workers: {args.workers}")
    print(f"Output : {cache_root}")
    print(f"Note   : each object will be randomly rotated around Z before rendering\n")

    ok = fail = 0

    if args.workers <= 1:
        for i, task in enumerate(tasks, 1):
            result = _process_one(task)
            if result is None:
                fail += 1
            else:
                ok += 1
            print(f"\r  {i}/{len(tasks)}  ok={ok}  fail={fail}", end="", flush=True)
    else:
        with mp.Pool(processes=args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_process_one, tasks), 1):
                if result is None:
                    fail += 1
                else:
                    ok += 1
                print(f"\r  {i}/{len(tasks)}  ok={ok}  fail={fail}", end="", flush=True)

    print(f"\n\nDone.  Saved: {ok}  Failed: {fail}")
    print("Cache layout:")
    for split in args.splits:
        pt_files = list((cache_root / split).rglob("*.pt"))
        print(f"  {cache_root / split}  ->  {len(pt_files):,} .pt files")


if __name__ == "__main__":
    main()
