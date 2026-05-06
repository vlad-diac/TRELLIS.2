"""
External Scaffold Bypass for TRELLIS.2

Replaces stage-one (stochastic sparse-structure sampling) with a deterministic
geometric scaffold derived from calibrated or assumed-orbit multi-view
silhouettes, a point cloud, or pre-computed depth maps.  The scaffold is
voxelised into TRELLIS.2 sparse coordinates and fed directly to stage-two
(shape SLAT), turning TRELLIS.2 into a high-quality geometry refiner and
texture synthesiser rather than a blind generative reconstructor.

Why this works
--------------
The conclusion analysis showed that stage-one is a *stochastic hypothesis
sampler* operating in an object-centric semantic latent frame.  It knows
nothing about camera poses and generates independent, complete canonical-
object completions per view.  Voxelwise fusion of these completions collapses
or duplicates geometry.

Bypassing stage-one with calibrated geometry eliminates the mode-selection
bottleneck entirely.  Stage-two (shape SLAT) and the decoder still contribute
substantial value: they refine the coarse scaffold into smooth, detailed dual-
grid geometry with complete PBR materials.

Modes
-----
visual-hull (default)
    Space-carving from multi-view silhouettes.  Silhouettes are extracted with
    rembg (the same model used internally by TRELLIS.2).  Camera poses can be
    provided as azimuths/elevations or left at the default 4-view orbit.  Works
    without any calibration data.  Ideal as a first experiment.

pointcloud
    Voxelise a PLY or PCD point cloud at the sparse-structure resolution.  Use
    this when you have SfM / LiDAR / MVS output for the scene.

depth
    Backproject per-image depth maps (e.g. from Depth Anything V2 or MoGe)
    into 3-D and voxelise.  Requires the depth scale and per-image camera
    azimuths/elevations.

Usage examples
--------------
# Visual hull from 4 images at evenly-spaced azimuths (elevation 15°)
python scripts/scaffold_bypass.py visual-hull \\
    front.png right.png rear.png left.png \\
    --elevation 15 --output-dir ./out

# Specify exact azimuths for non-uniform views
python scripts/scaffold_bypass.py visual-hull \\
    img1.png img2.png img3.png \\
    --azimuths 0 45 200 --elevation 20 --output-dir ./out

# Point cloud bypass
python scripts/scaffold_bypass.py pointcloud \\
    scan.ply --reference-image front.png --output-dir ./out

# Depth-map bypass (one depth PNG per image)
python scripts/scaffold_bypass.py depth \\
    front.png --depth-map depth_front.png \\
    --azimuth 0 --elevation 0 --output-dir ./out

# Combine scaffold bypass with condition fusion (P2+P1)
python scripts/scaffold_bypass.py visual-hull \\
    front.png right.png rear.png left.png \\
    --cond-fusion mean --output-dir ./out

# Skip texture (sparse-only mode for fast scaffold inspection)
python scripts/scaffold_bypass.py visual-hull \\
    front.png rear.png \\
    --scaffold-only --output-dir ./out
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"


def print_section(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Camera / projection utilities
# ---------------------------------------------------------------------------

def make_look_at_rotation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """
    Return a 3×3 world-to-camera rotation matrix for a camera placed on a unit
    sphere at (azimuth, elevation) looking toward the world origin.

    Convention: z_cam points from origin toward the camera (i.e. the camera
    looks along -z_cam), x_cam is right, y_cam is up.

    Args:
        azimuth_deg: Azimuth around world-y in degrees (0 = front = +z_world).
        elevation_deg: Elevation above the equatorial plane in degrees.

    Returns:
        (3, 3) float64 rotation matrix mapping world → camera.
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Camera position on unit sphere
    cx = np.cos(el) * np.sin(az)
    cy = np.sin(el)
    cz = np.cos(el) * np.cos(az)
    z_cam = np.array([cx, cy, cz])
    z_cam /= np.linalg.norm(z_cam)

    # Camera right vector: cross(world_up, z_cam)
    world_up = np.array([0.0, 1.0, 0.0])
    x_cam = np.cross(world_up, z_cam)
    if np.linalg.norm(x_cam) < 1e-8:
        # Degenerate: camera at pole — use world_z as up
        world_up = np.array([0.0, 0.0, 1.0])
        x_cam = np.cross(world_up, z_cam)
    x_cam /= np.linalg.norm(x_cam)

    # Camera up: cross(z_cam, x_cam)
    y_cam = np.cross(z_cam, x_cam)

    # Row-wise world-to-camera rotation
    return np.stack([x_cam, y_cam, z_cam], axis=0).astype(np.float64)


def project_orthographic(
    points_world: np.ndarray,   # (N, 3)
    R: np.ndarray,              # (3, 3) world-to-camera
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project world points via orthographic projection.

    Returns ``(u, v)`` coordinates in the approximate object-space range
    ``[-0.5, 0.5]`` that can be directly compared to silhouette masks
    scaled to image size.
    """
    pts_cam = (R @ points_world.T).T   # (N, 3)
    return pts_cam[:, 0], pts_cam[:, 1]


# ---------------------------------------------------------------------------
# Space carving (visual hull)
# ---------------------------------------------------------------------------

def extract_silhouette(image: Image.Image, pipeline) -> np.ndarray:
    """
    Extract a binary silhouette mask from a PIL image using the rembg model
    already loaded in the pipeline.

    Returns a ``(H, W)`` bool array — True where the object is present.
    """
    # Use the pipeline's preprocess_image which runs rembg + crops
    processed = pipeline.preprocess_image(image)
    arr = np.array(processed.convert("RGBA"))
    # Alpha channel > 0 indicates the object
    return arr[:, :, 3] > 0


def space_carve(
    masks: List[np.ndarray],     # list of (H, W) bool arrays
    rotations: List[np.ndarray], # list of (3, 3) world-to-camera matrices
    grid_size: int = 32,
    min_views: int = 1,
) -> np.ndarray:
    """
    Visual-hull space carving.

    For each voxel in a ``grid_size^3`` grid covering the unit cube
    ``[-0.5, 0.5]^3``, project to each camera view and check whether the
    projected pixel falls inside the silhouette mask.  A voxel is kept if it
    projects inside the silhouette in at least ``min_views`` views.

    ``min_views = 1`` gives the union hull (superset of true geometry).
    ``min_views = len(masks)`` gives the intersection hull (visual hull proper,
    closest to true geometry but may miss thin parts that are partially
    occluded).

    Args:
        masks: Binary silhouette masks, one per view.
        rotations: World-to-camera rotation matrices, one per view.
        grid_size: Voxel grid resolution (should match the sparse-structure
            resolution, typically 32).
        min_views: Minimum number of views a voxel must be inside to survive.

    Returns:
        Boolean (G, G, G) occupancy array.
    """
    G = grid_size
    step = 1.0 / G
    coords_1d = np.linspace(-0.5 + step / 2, 0.5 - step / 2, G, dtype=np.float64)
    xg, yg, zg = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    points = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=-1)  # (G^3, 3)

    votes = np.zeros(G * G * G, dtype=np.int32)

    for mask, R in zip(masks, rotations):
        H, W = mask.shape
        u, v = project_orthographic(points, R)

        # Map [-0.5, 0.5] → pixel indices [0, W-1] / [0, H-1]
        ui = np.clip(np.round((u + 0.5) * W - 0.5).astype(np.int32), 0, W - 1)
        vi = np.clip(np.round((v + 0.5) * H - 0.5).astype(np.int32), 0, H - 1)

        in_sil = mask[vi, ui].astype(np.int32)
        votes += in_sil

    occupied = votes >= max(1, min_views)
    return occupied.reshape(G, G, G)


# ---------------------------------------------------------------------------
# Point cloud voxelisation
# ---------------------------------------------------------------------------

def voxelize_pointcloud(
    path: str,
    grid_size: int = 32,
    aabb: Optional[Tuple] = None,
) -> np.ndarray:
    """
    Voxelise a PLY or PCD point cloud at ``grid_size`` resolution.

    The bounding box is normalised to ``[-0.5, 0.5]^3`` (or an explicit AABB
    can be provided).  Each point votes for its enclosing voxel; voxels with
    ≥1 vote are occupied.

    Requires ``open3d`` (``pip install open3d``).

    Args:
        path: Path to the point cloud file.
        grid_size: Voxel grid resolution.
        aabb: Optional ``((xmin, ymin, zmin), (xmax, ymax, zmax))`` AABB for
            normalisation.  Defaults to the point cloud's own bounding box.

    Returns:
        Boolean ``(G, G, G)`` occupancy array.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for point cloud mode. "
            "Install it with: pip install open3d"
        )

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)

    if aabb is None:
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
    else:
        lo = np.array(aabb[0], dtype=np.float64)
        hi = np.array(aabb[1], dtype=np.float64)

    extent = hi - lo
    extent = np.where(extent < 1e-8, 1.0, extent)

    # Normalize to [0, 1] then shift to [-0.5, 0.5]
    pts_norm = (pts - lo) / extent - 0.5   # (N, 3) in [-0.5, 0.5]

    G = grid_size
    xi = np.clip(np.floor((pts_norm[:, 0] + 0.5) * G).astype(np.int32), 0, G - 1)
    yi = np.clip(np.floor((pts_norm[:, 1] + 0.5) * G).astype(np.int32), 0, G - 1)
    zi = np.clip(np.floor((pts_norm[:, 2] + 0.5) * G).astype(np.int32), 0, G - 1)

    vol = np.zeros((G, G, G), dtype=bool)
    vol[xi, yi, zi] = True
    return vol


# ---------------------------------------------------------------------------
# Depth-map voxelisation
# ---------------------------------------------------------------------------

def voxelize_depth_maps(
    depth_maps: List[np.ndarray],
    azimuths: List[float],
    elevations: List[float],
    grid_size: int = 32,
    depth_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Voxelise depth maps from multiple views into a single occupancy grid using
    orthographic projection and the same camera model as the silhouette carver.

    Each depth map is backprojected to a set of approximate 3-D surface points,
    which are accumulated into a vote grid.  Voxels receiving ≥1 vote survive.

    Args:
        depth_maps: List of ``(H, W)`` float arrays with larger values farther.
        azimuths: Camera azimuth per depth map, in degrees.
        elevations: Camera elevation per depth map, in degrees.
        grid_size: Voxel grid resolution.
        depth_range: Optional ``(min_d, max_d)`` for normalisation.

    Returns:
        Boolean ``(G, G, G)`` occupancy array.
    """
    G = grid_size
    step = 1.0 / G
    acc = np.zeros((G, G, G), dtype=np.int32)

    for depth, az, el in zip(depth_maps, azimuths, elevations):
        H, W = depth.shape
        d = depth.astype(np.float64)
        d_min = d.min() if depth_range is None else depth_range[0]
        d_max = d.max() if depth_range is None else depth_range[1]
        if d_max > d_min:
            d = (d - d_min) / (d_max - d_min)
        else:
            d = np.zeros_like(d)

        # Image coords → object-space [-0.5, 0.5]
        uu = np.arange(W, dtype=np.float64)
        vv = np.arange(H, dtype=np.float64)
        u2d, v2d = np.meshgrid(uu, vv)
        xc = u2d / W - 0.5          # camera-space x (lateral)
        yc = -(v2d / H - 0.5)       # camera-space y (vertical, flipped)
        zc = 1.0 - d                # toward camera = high z

        # Rotate from camera frame to world frame (transpose of R)
        R = make_look_at_rotation(az, el)
        pts_cam = np.stack([xc.ravel(), yc.ravel(), zc.ravel()], axis=-1)  # (N, 3)
        pts_world = (R.T @ pts_cam.T).T  # (N, 3)

        xi = np.clip(np.floor((pts_world[:, 0] + 0.5) * G).astype(np.int32), 0, G - 1)
        yi = np.clip(np.floor((pts_world[:, 1] + 0.5) * G).astype(np.int32), 0, G - 1)
        zi = np.clip(np.floor((pts_world[:, 2] + 0.5) * G).astype(np.int32), 0, G - 1)
        np.add.at(acc, (xi, yi, zi), 1)

    return acc > 0


# ---------------------------------------------------------------------------
# Scaffold → TRELLIS.2 coords
# ---------------------------------------------------------------------------

def occupancy_to_coords(occupancy: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Convert a boolean ``(G, G, G)`` occupancy array into a TRELLIS.2
    sparse-coordinate tensor of shape ``(N, 4)`` [batch_idx=0, x, y, z].
    """
    xi, yi, zi = np.where(occupancy)
    batch = np.zeros_like(xi)
    coords = np.stack([batch, xi, yi, zi], axis=-1).astype(np.int32)
    return torch.from_numpy(coords).to(device)


# ---------------------------------------------------------------------------
# Main pipeline bypass
# ---------------------------------------------------------------------------

def run_scaffold_bypass(
    pipeline: "Trellis2ImageTo3DPipeline",
    occupancy: np.ndarray,
    conditioning_images: List[Image.Image],
    cond_fusion_mode: str = "primary",
    scaffold_only: bool = False,
    pipeline_type: Optional[str] = None,
    max_num_tokens: int = 49152,
    seed: int = 42,
    shape_slat_sampler_params: dict = {},
    tex_slat_sampler_params: dict = {},
) -> dict:
    """
    Run TRELLIS.2 stages 2–4 (shape SLAT, texture SLAT, decode) starting from
    an externally computed occupancy grid.

    Stage 1 (stochastic sparse-structure sampling) is skipped entirely.  The
    scaffold occupancy is voxelised into TRELLIS.2 sparse coordinates and passed
    directly to ``sample_shape_slat()`` / ``sample_shape_slat_cascade()``.

    Args:
        pipeline: Loaded ``Trellis2ImageTo3DPipeline``.
        occupancy: Boolean ``(G, G, G)`` array from space carving / point cloud.
        conditioning_images: PIL images used for SLAT conditioning.  When
            ``cond_fusion_mode="primary"``, only the first image is used.  When
            ``cond_fusion_mode`` is ``"mean"`` or ``"concat"``, all images are
            fused with ``get_cond_multi()``.
        cond_fusion_mode: How to fuse per-view SLAT conditioning tokens.
            ``"primary"`` (default), ``"mean"``, or ``"concat"``.
        scaffold_only: Skip shape/texture SLAT and return just the scaffold
            coords (for inspection).
        pipeline_type: ``'512'``, ``'1024'``, ``'1024_cascade'``, or
            ``'1536_cascade'``.
        max_num_tokens: Token budget for cascade upsampling.
        seed: Random seed.
        shape_slat_sampler_params: Extra kwargs for shape SLAT sampling.
        tex_slat_sampler_params: Extra kwargs for texture SLAT sampling.

    Returns:
        dict with keys:
            ``"coords"``    — TRELLIS.2 coordinate tensor (N, 4)
            ``"occupancy"`` — input boolean occupancy grid
            ``"meshes"``    — ``List[MeshWithVoxel]`` (empty if scaffold_only)
    """
    torch.manual_seed(seed)
    pipeline_type = pipeline_type or pipeline.default_pipeline_type

    n_voxels = int(occupancy.sum())
    print(f"[scaffold_bypass] scaffold has {n_voxels} occupied voxels "
          f"({100 * n_voxels / occupancy.size:.1f}% density)")

    coords = occupancy_to_coords(occupancy, device=pipeline.device)

    if scaffold_only:
        return {"coords": coords, "occupancy": occupancy, "meshes": []}

    # --- Conditioning ---
    if cond_fusion_mode == "primary":
        cond_512 = pipeline.get_cond([conditioning_images[0]], 512)
        cond_1024 = (pipeline.get_cond([conditioning_images[0]], 1024)
                     if pipeline_type != '512' else None)
    else:
        cond_512 = pipeline.get_cond_multi(conditioning_images, 512, cond_fusion_mode)
        cond_1024 = (pipeline.get_cond_multi(conditioning_images, 1024, cond_fusion_mode)
                     if pipeline_type != '512' else None)

    # Primary conditioning for texture (always single-view)
    tex_cond_512 = pipeline.get_cond([conditioning_images[0]], 512)
    tex_cond_1024 = (pipeline.get_cond([conditioning_images[0]], 1024)
                     if pipeline_type != '512' else None)

    # --- Shape SLAT (stages 2a / 2b) ---
    if pipeline_type == '512':
        shape_slat = pipeline.sample_shape_slat(
            cond_512,
            pipeline.models['shape_slat_flow_model_512'],
            coords, shape_slat_sampler_params,
        )
        tex_slat = pipeline.sample_tex_slat(
            tex_cond_512,
            pipeline.models['tex_slat_flow_model_512'],
            shape_slat, tex_slat_sampler_params,
        )
        res = 512

    elif pipeline_type == '1024':
        shape_slat = pipeline.sample_shape_slat(
            cond_1024,
            pipeline.models['shape_slat_flow_model_1024'],
            coords, shape_slat_sampler_params,
        )
        tex_slat = pipeline.sample_tex_slat(
            tex_cond_1024,
            pipeline.models['tex_slat_flow_model_1024'],
            shape_slat, tex_slat_sampler_params,
        )
        res = 1024

    elif pipeline_type == '1024_cascade':
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models['shape_slat_flow_model_512'],
            pipeline.models['shape_slat_flow_model_1024'],
            512, 1024, coords, shape_slat_sampler_params, max_num_tokens,
        )
        tex_slat = pipeline.sample_tex_slat(
            tex_cond_1024,
            pipeline.models['tex_slat_flow_model_1024'],
            shape_slat, tex_slat_sampler_params,
        )

    elif pipeline_type == '1536_cascade':
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models['shape_slat_flow_model_512'],
            pipeline.models['shape_slat_flow_model_1024'],
            512, 1536, coords, shape_slat_sampler_params, max_num_tokens,
        )
        tex_slat = pipeline.sample_tex_slat(
            tex_cond_1024,
            pipeline.models['tex_slat_flow_model_1024'],
            shape_slat, tex_slat_sampler_params,
        )
    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type!r}")

    torch.cuda.empty_cache()
    meshes = pipeline.decode_latent(shape_slat, tex_slat, res)
    return {"coords": coords, "occupancy": occupancy, "meshes": meshes}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="External scaffold bypass for TRELLIS.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # --- visual-hull mode ---
    vh = sub.add_parser("visual-hull", help="Space-carving from multi-view silhouettes")
    vh.add_argument("images", nargs="+", help="Input image paths (2+ for best results)")
    vh.add_argument("--azimuths", nargs="+", type=float, default=None,
                    help="Camera azimuths in degrees (default: evenly spaced over 360°)")
    vh.add_argument("--elevation", type=float, default=15.0,
                    help="Camera elevation in degrees (default: 15)")
    vh.add_argument("--min-views", type=int, default=None,
                    help="Min views a voxel must be inside to survive "
                         "(default: all views = visual hull proper)")

    # --- pointcloud mode ---
    pc = sub.add_parser("pointcloud", help="Voxelise a PLY / PCD point cloud")
    pc.add_argument("pointcloud", help="Path to .ply or .pcd file")
    pc.add_argument("--reference-image", required=True,
                    help="Reference image for SLAT conditioning")
    pc.add_argument("--extra-images", nargs="*", default=[],
                    help="Additional images for multi-view SLAT conditioning")

    # --- depth mode ---
    dep = sub.add_parser("depth", help="Voxelise from monocular depth maps")
    dep.add_argument("images", nargs="+", help="Input image paths")
    dep.add_argument("--depth-maps", nargs="+", required=True,
                     help="Depth map paths (one per image, PNG/EXR, larger = farther)")
    dep.add_argument("--azimuths", nargs="+", type=float, required=True,
                     help="Camera azimuth per image in degrees")
    dep.add_argument("--elevations", nargs="+", type=float, default=None,
                     help="Camera elevation per image in degrees (default: 0 for all)")

    # --- shared args ---
    for sp in (vh, pc, dep):
        sp.add_argument("--output-dir", default="./out",
                        help="Directory for output GLB files (default: ./out)")
        sp.add_argument("--grid-size", type=int, default=32,
                        help="Voxel grid resolution (default: 32, must match "
                             "sparse-structure model resolution)")
        sp.add_argument("--cond-fusion", default="primary",
                        choices=["primary", "mean", "concat"],
                        help="How to fuse per-view SLAT conditioning tokens "
                             "(default: primary = first image only)")
        sp.add_argument("--pipeline-type", default=None,
                        choices=["512", "1024", "1024_cascade", "1536_cascade"],
                        help="Pipeline resolution type")
        sp.add_argument("--scaffold-only", action="store_true",
                        help="Print scaffold stats and exit; skip shape/tex SLAT")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--model", default="microsoft/TRELLIS.2-4B",
                        help="HuggingFace model ID or local path")
        sp.add_argument("--symmetry-axis", type=int, default=None,
                        choices=[0, 1, 2],
                        help="Apply port/starboard symmetry prior on this axis "
                             "(0=x, 1=y, 2=z) before passing coords to stage 2")
        sp.add_argument("--decimation-target", type=int, default=500_000,
                        help="GLB mesh decimation target (default: 500000)")
        sp.add_argument("--texture-size", type=int, default=1024,
                        help="GLB texture atlas size (default: 1024)")

    return p.parse_args()


def save_output(pipeline, result: dict, args: argparse.Namespace, label: str) -> None:
    """Save scaffold stats and optionally the GLB output."""
    import o_voxel

    os.makedirs(args.output_dir, exist_ok=True)
    ts = timestamp()

    occ = result["occupancy"]
    coords = result["coords"]
    print(f"\n[scaffold_bypass] scaffold stats ({label}):")
    print(f"  Grid size      : {occ.shape[0]}³")
    print(f"  Occupied voxels: {int(occ.sum())} / {occ.size} "
          f"({100 * occ.sum() / occ.size:.2f}%)")
    print(f"  Coord tensor   : {tuple(coords.shape)}")

    for i, mesh in enumerate(result["meshes"]):
        glb_path = os.path.join(
            args.output_dir,
            f"{ts}_{label}_mesh{i}.glb",
        )
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=getattr(args, "grid_size", 32),
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        glb.export(glb_path, extension_webp=True)
        print(f"  GLB saved: {glb_path}")


def main() -> None:
    args = parse_args()

    print_section(f"TRELLIS.2 Scaffold Bypass — mode: {args.mode}")

    # --- Load pipeline ---
    if not args.scaffold_only:
        print(f"[scaffold_bypass] loading pipeline: {args.model} ...")
        t0 = time.time()
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
        pipeline.cuda()
        print(f"[scaffold_bypass] pipeline loaded in {time.time() - t0:.1f}s")
    else:
        pipeline = None

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Build occupancy grid ---
    print_section("Building occupancy scaffold")

    if args.mode == "visual-hull":
        images = [Image.open(p).convert("RGB") for p in args.images]
        n = len(images)

        # Camera azimuths
        if args.azimuths is not None:
            if len(args.azimuths) != n:
                raise ValueError(
                    f"--azimuths has {len(args.azimuths)} values but "
                    f"{n} images were provided."
                )
            azimuths = args.azimuths
        else:
            azimuths = [360.0 * i / n for i in range(n)]
            print(f"[visual-hull] auto azimuths: {[f'{a:.1f}°' for a in azimuths]}")

        elevations = [args.elevation] * n
        min_views = args.min_views if args.min_views is not None else n

        # Extract silhouettes
        print(f"[visual-hull] extracting silhouettes for {n} image(s) ...")
        if pipeline is not None:
            masks = [extract_silhouette(img, pipeline) for img in images]
        else:
            # Scaffold-only mode: build a dummy pipeline just for rembg
            _pl = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
            masks = [extract_silhouette(img, _pl) for img in images]
            del _pl

        rotations = [make_look_at_rotation(az, el)
                     for az, el in zip(azimuths, elevations)]

        print(f"[visual-hull] space carving (grid={args.grid_size}³, "
              f"min_views={min_views}) ...")
        t0 = time.time()
        occupancy = space_carve(masks, rotations, args.grid_size, min_views)
        print(f"[visual-hull] carving done in {time.time() - t0:.2f}s — "
              f"{int(occupancy.sum())} voxels")

        conditioning_images = images

    elif args.mode == "pointcloud":
        print(f"[pointcloud] voxelising {args.pointcloud} ...")
        occupancy = voxelize_pointcloud(args.pointcloud, args.grid_size)
        ref_img = Image.open(args.reference_image).convert("RGB")
        extra_imgs = [Image.open(p).convert("RGB") for p in args.extra_images]
        conditioning_images = [ref_img] + extra_imgs

    elif args.mode == "depth":
        images = [Image.open(p).convert("RGB") for p in args.images]
        depth_maps = []
        for dpath in args.depth_maps:
            da = np.array(Image.open(dpath).convert("L")).astype(np.float32)
            depth_maps.append(da)

        azimuths = args.azimuths
        elevations = (args.elevations if args.elevations is not None
                      else [0.0] * len(images))

        print(f"[depth] voxelising {len(depth_maps)} depth map(s) ...")
        occupancy = voxelize_depth_maps(
            depth_maps, azimuths, elevations, args.grid_size
        )
        conditioning_images = images

    # --- Optional symmetry prior ---
    if args.symmetry_axis is not None:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline as _P
        print(f"[scaffold_bypass] applying symmetry prior on axis {args.symmetry_axis} ...")
        before = int(occupancy.sum())
        # Convert occupancy to coords, apply symmetry, convert back
        tmp_coords = occupancy_to_coords(occupancy, device="cpu")
        sym_coords = _P.apply_symmetry_prior(
            tmp_coords, grid_size=args.grid_size, axis=args.symmetry_axis
        )
        # Rehydrate into occupancy grid
        occ_sym = np.zeros((args.grid_size,) * 3, dtype=bool)
        c = sym_coords.numpy()
        occ_sym[c[:, 1], c[:, 2], c[:, 3]] = True
        occupancy = occ_sym
        print(f"[scaffold_bypass] symmetry: {before} → {int(occupancy.sum())} voxels")

    # --- Run pipeline bypass or scaffold-only report ---
    if args.scaffold_only:
        coords = occupancy_to_coords(occupancy, device="cpu")
        result = {"coords": coords, "occupancy": occupancy, "meshes": []}
        save_output(None, result, args, label=args.mode)
        return

    print_section("Running TRELLIS.2 stages 2–4 on scaffold")
    t0 = time.time()
    result = run_scaffold_bypass(
        pipeline=pipeline,
        occupancy=occupancy,
        conditioning_images=conditioning_images,
        cond_fusion_mode=args.cond_fusion,
        scaffold_only=False,
        pipeline_type=args.pipeline_type,
        seed=args.seed,
        shape_slat_sampler_params={},
        tex_slat_sampler_params={},
    )
    print(f"[scaffold_bypass] stage 2–4 done in {time.time() - t0:.1f}s")

    save_output(pipeline, result, args, label=args.mode)
    print_section("Done")


if __name__ == "__main__":
    main()
