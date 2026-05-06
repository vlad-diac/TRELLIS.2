from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @staticmethod
    def _fuse_slat_feats(slats: list, mode: str) -> "SparseTensor":
        """
        Fuse a list of per-view SparseTensors (sharing identical coords) by
        combining their feature vectors with the requested strategy.

        Args:
            slats (list[SparseTensor]): Per-view SLAT tensors, all with the
                same coords layout.
            mode (str): One of ``"slat-mean"``, ``"slat-norm-weighted"``,
                ``"slat-max"``.

        Returns:
            SparseTensor: Single tensor with the same coords as ``slats[0]``
                and fused feature vectors.
        """
        stacked = torch.stack([s.feats for s in slats], dim=0)  # (V, T, C)

        if mode == "slat-mean":
            fused = stacked.mean(dim=0)
        elif mode == "slat-norm-weighted":
            # Weight each view's contribution by softmax(||feat||_2) per token.
            # Views that generate high-magnitude features at a coordinate are
            # more "confident" about the geometry there.
            norms = stacked.norm(dim=-1, keepdim=True)      # (V, T, 1)
            weights = torch.softmax(norms, dim=0)           # (V, T, 1)
            fused = (stacked * weights).sum(dim=0)          # (T, C)
        elif mode == "slat-max":
            fused = stacked.max(dim=0).values               # (T, C)
        else:
            raise ValueError(
                f"Unknown slat_fusion_mode: {mode!r}. "
                f"Choose 'slat-mean', 'slat-norm-weighted', or 'slat-max'."
            )

        return slats[0].replace(fused)

    def sample_shape_slat_multi(
        self,
        conds_per_view: List[dict],
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        slat_fusion_mode: str = "slat-mean",
    ) -> "SparseTensor":
        """
        Run the shape SLAT flow model once per view using a shared coordinate
        set, then fuse the resulting feature vectors.

        Each view generates a full 32-channel feature tensor at every active
        sparse coordinate.  Fusing these tensors combines geometric evidence
        from all viewpoints before decoding, operating at the true geometry
        representation level rather than the occupancy voxel level.

        Args:
            conds_per_view (List[dict]): Per-view conditioning dicts from
                ``get_cond()``.  All views share the same ``coords`` so the
                resulting SparseTensors have identical coordinate layouts.
            flow_model: The shape SLAT flow model to use.
            coords (torch.Tensor): Sparse coordinate tensor of shape (T, 4)
                [batch_idx, x, y, z] — shared across all views.
            sampler_params (dict): Extra sampler kwargs (merged with
                ``self.shape_slat_sampler_params`` inside
                ``sample_shape_slat``).
            slat_fusion_mode (str):
                ``"slat-mean"``          — average features across views.
                ``"slat-norm-weighted"`` — weight by per-token feature norm.
                ``"slat-max"``           — element-wise maximum across views.

        Returns:
            SparseTensor: Fused shape SLAT with the same coords as input.
        """
        per_view_slats = []
        for view_idx, cond in enumerate(conds_per_view):
            print(f"[slat-fusion] sampling shape SLAT view "
                  f"{view_idx + 1}/{len(conds_per_view)} ({slat_fusion_mode}) ...")
            slat = self.sample_shape_slat(cond, flow_model, coords, sampler_params)
            per_view_slats.append(slat)

        fused = self._fuse_slat_feats(per_view_slats, slat_fusion_mode)
        print(f"[slat-fusion] fused {len(conds_per_view)} SLAT views "
              f"→ {fused.feats.shape[0]:,} tokens × {fused.feats.shape[1]}ch")
        return fused

    def sample_shape_slat_cascade_multi(
        self,
        lr_cond: dict,
        conds_per_view: List[dict],
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        slat_fusion_mode: str = "slat-mean",
    ) -> tuple:
        """
        Cascade SLAT sampling with per-view HR-stage fusion.

        The LR stage uses ``lr_cond`` (primary view) to produce a coarse SLAT
        which is upsampled to generate HR coordinates.  The HR stage is then
        run once per view in ``conds_per_view`` and the resulting feature
        vectors are fused before returning.

        Args:
            lr_cond (dict): Conditioning dict for the LR flow stage (primary
                view, 512-px).
            conds_per_view (List[dict]): Per-view conditioning dicts for the
                HR flow stage (one per view, 1024-px).
            flow_model_lr: LR shape SLAT flow model.
            flow_model: HR shape SLAT flow model.
            lr_resolution (int): LR coordinate grid size (typically 512).
            resolution (int): Target HR resolution (1024 or 1536).
            coords (torch.Tensor): Coarse-stage sparse coords (T, 4).
            sampler_params (dict): Extra sampler kwargs.
            max_num_tokens (int): Token budget cap for HR coords.
            slat_fusion_mode (str): Feature fusion mode for the HR stage.

        Returns:
            Tuple[SparseTensor, int]: Fused HR SLAT and effective resolution.
        """
        # --- LR stage: primary view only (unchanged from existing cascade) ---
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        merged_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat_lr = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **merged_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (LR, primary)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat_lr.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat_lr.device)
        slat_lr = slat_lr * std + mean

        # --- Upsample LR SLAT to get HR coordinate proposals ---
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat_lr, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False

        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution "
                          f"is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128

        # --- HR stage: run per-view, fuse feature vectors ---
        shape_slat = self.sample_shape_slat_multi(
            conds_per_view, flow_model, coords, sampler_params, slat_fusion_mode,
        )
        return shape_slat, hr_resolution

    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def _sample_occupancy_volume(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        samples_per_view: int = 1,
        return_logits: bool = False,
    ):
        """
        Run the sparse structure flow model and return the decoded boolean
        occupancy volume.  Used as a building block for multi-view fusion.

        When ``samples_per_view > 1`` the flow model is run that many times
        with independent noise seeds and the raw decoder logits are averaged
        before thresholding at 0.  This intra-view ensemble suppresses
        per-run hallucinations and produces a more reliable single-view
        occupancy prediction.

        Args:
            cond (dict): Conditioning dict from ``get_cond()``.
            num_samples (int): Batch dimension (typically 1).
            sampler_params (dict): Extra sampler kwargs.
            samples_per_view (int): Number of independent diffusion runs to
                average *before* thresholding (default 1 = no ensemble).
            return_logits (bool): When True, return a
                ``(decoded_bool, mean_logits)`` tuple where ``mean_logits`` is
                the raw float tensor of shape (B, 1, reso, reso, reso) before
                the ``> 0`` threshold is applied.  Default False preserves the
                original single-return behaviour.

        Returns:
            torch.Tensor: Boolean tensor of shape (B, 1, reso, reso, reso), or
            a ``(bool_tensor, float_tensor)`` tuple when ``return_logits=True``.
        """
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        merged_params = {**self.sparse_structure_sampler_params, **sampler_params}

        if self.low_vram:
            flow_model.to(self.device)

        raw_logits_list = []
        for run_idx in range(samples_per_view):
            noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **merged_params,
                verbose=True,
                tqdm_desc=f"Sampling sparse structure (per view, run {run_idx + 1}/{samples_per_view})",
            ).samples

            decoder = self.models['sparse_structure_decoder']
            if self.low_vram:
                flow_model.cpu()
                decoder.to(self.device)

            raw_logits = decoder(z_s)  # (B, 1, reso, reso, reso) float logits
            raw_logits_list.append(raw_logits)

            if self.low_vram:
                decoder.cpu()
                flow_model.to(self.device)

        if self.low_vram:
            flow_model.cpu()

        # Average logits across ensemble runs, then threshold at 0
        mean_logits = torch.stack(raw_logits_list, dim=0).mean(dim=0)
        decoded = mean_logits > 0  # (B, 1, reso, reso, reso) bool
        if return_logits:
            return decoded, mean_logits
        return decoded

    @staticmethod
    def _filter_isolated_voxels(vol: torch.Tensor, min_neighbors: int) -> torch.Tensor:
        """
        Remove voxels that have fewer than ``min_neighbors`` occupied neighbors
        in their 3×3×3 neighbourhood.  Applied after cross-view fusion to
        discard floating hallucination islands while preserving contiguous
        geometry.

        Args:
            vol (torch.Tensor): Boolean tensor (B, 1, X, Y, Z).
            min_neighbors (int): Minimum required occupied neighbours (1–26).

        Returns:
            torch.Tensor: Filtered boolean tensor, same shape.
        """
        if min_neighbors <= 0:
            return vol
        import torch.nn.functional as F
        kernel = torch.ones(1, 1, 3, 3, 3, dtype=torch.float32, device=vol.device)
        # Count all occupied voxels in each 3×3×3 window (including self)
        neighbor_count = F.conv3d(vol.float(), kernel, padding=1)
        # Subtract self so we count only neighbours
        neighbor_count = neighbor_count - vol.float()
        return vol & (neighbor_count >= min_neighbors)

    @staticmethod
    def _smooth_logit_volume(logits: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply a 3-D Gaussian blur to a float logit volume before thresholding.

        Voxels with weak scores are reinforced by strongly-occupied neighbours,
        implementing "probabilistic spatial diffusion".  The kernel is built
        analytically so no learned weights are required.

        Args:
            logits (torch.Tensor): Float tensor of shape (B, 1, X, Y, Z).
            sigma (float): Gaussian standard deviation in voxels.  Values of
                0.5–1.5 give useful smoothing; 0.0 is a no-op identity.

        Returns:
            torch.Tensor: Smoothed float tensor, same shape.
        """
        if sigma <= 0.0:
            return logits
        import torch.nn.functional as F
        import math

        radius = math.ceil(2.0 * sigma)
        size = 2 * radius + 1

        # Build a 1-D Gaussian, then outer-product to 3-D
        coords_1d = torch.arange(size, dtype=torch.float32, device=logits.device) - radius
        gauss_1d = torch.exp(-0.5 * (coords_1d / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()

        kernel = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
        kernel = kernel.view(1, 1, size, size, size)

        return F.conv3d(logits.float(), kernel, padding=radius)

    def sample_sparse_structure_multi(
        self,
        conds_per_view: List[dict],
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        fusion_mode: str = "union",
        vote_threshold: float = 0.5,
        logit_threshold: float = 0.0,
        logit_smooth_sigma: float = 0.0,
        samples_per_view: int = 1,
        filter_min_neighbors: int = 0,
    ) -> torch.Tensor:
        """
        Run the sparse structure flow model once per view, fuse the resulting
        occupancy volumes, and return unified sparse coords.

        Binary fusion modes (``union``, ``vote``) threshold each view's
        occupancy **before** cross-view merging.  Logit fusion modes
        (``logit-mean``, ``logit-max``, ``logit-sum``) preserve the raw
        decoder float scores across all views and apply a **single** threshold
        after fusion, retaining weak-evidence voxels that would otherwise be
        discarded by per-view thresholding.

        Args:
            conds_per_view (List[dict]): One conditioning dict per input image,
                as returned by ``get_cond()``.
            resolution (int): Target coord resolution (e.g. 32 for 1024_cascade).
            num_samples (int): Batch size per view (typically 1).
            sampler_params (dict): Extra sampler kwargs forwarded to each run.
            fusion_mode (str):
                ``"union"``      — occupied if *any* view votes yes (binary).
                ``"vote"``       — occupied if fraction >= ``vote_threshold``
                                   (binary).
                ``"logit-mean"`` — fuse raw logits by averaging across views,
                                   then threshold at ``logit_threshold``.
                ``"logit-max"``  — fuse by taking the per-voxel maximum logit,
                                   then threshold at ``logit_threshold``.
                ``"logit-sum"``  — fuse by summing logits across views, then
                                   threshold at ``logit_threshold``.
            vote_threshold (float): Fraction threshold for ``"vote"`` mode.
            logit_threshold (float): Threshold applied to the fused logit
                volume in logit-* modes (default 0.0 ≈ decoder decision
                boundary).
            logit_smooth_sigma (float): If > 0, apply a 3-D Gaussian blur with
                this sigma (in voxels) to the fused logit volume before
                thresholding.  Reinforces weak-evidence regions surrounded by
                stronger neighbours.  0.0 = disabled (default).
            samples_per_view (int): Number of independent diffusion runs
                averaged *within* each view before cross-view fusion.
                Values of 3–5 reduce per-view hallucinations at the cost of
                proportionally more compute.  Default is 1 (no ensemble).
            filter_min_neighbors (int): After fusion, remove voxels with fewer
                than this many occupied neighbours in their 3×3×3 block.
                0 = disabled (default).  2–4 is a good range for cleaning
                isolated hallucination islands without eroding real geometry.

        Returns:
            torch.Tensor: Fused coords tensor of shape (N_occupied, 4)
                [batch_idx, x, y, z].
        """
        _LOGIT_MODES = {"logit-mean", "logit-max", "logit-sum"}
        use_logit_fusion = fusion_mode in _LOGIT_MODES

        per_view_vols = []   # bool tensors for binary modes
        per_view_logits = [] # float tensors for logit modes

        for view_idx, cond in enumerate(conds_per_view):
            print(f"[multi-fusion] sampling view {view_idx + 1}/{len(conds_per_view)}"
                  + (f" (×{samples_per_view} ensemble)" if samples_per_view > 1 else "") + " ...")
            if use_logit_fusion:
                vol, logits = self._sample_occupancy_volume(
                    cond, num_samples, sampler_params, samples_per_view, return_logits=True
                )
                per_view_logits.append(logits)
                per_view_vols.append(vol)
            else:
                vol = self._sample_occupancy_volume(cond, num_samples, sampler_params, samples_per_view)
                per_view_vols.append(vol)
            print(f"[multi-fusion] view {view_idx + 1} occupied voxels: {per_view_vols[-1].sum().item()}")

        if use_logit_fusion:
            # Stack logits → (n_views, B, 1, reso, reso, reso)
            stacked_logits = torch.stack(per_view_logits, dim=0)

            if fusion_mode == "logit-mean":
                fused_logits = stacked_logits.mean(dim=0)
            elif fusion_mode == "logit-max":
                fused_logits = stacked_logits.max(dim=0).values
            elif fusion_mode == "logit-sum":
                fused_logits = stacked_logits.sum(dim=0)

            if logit_smooth_sigma > 0.0:
                before_smooth = int((fused_logits > logit_threshold).sum().item())
                fused_logits = self._smooth_logit_volume(fused_logits, logit_smooth_sigma)
                after_smooth = int((fused_logits > logit_threshold).sum().item())
                print(f"[multi-fusion] logit smoothing (sigma={logit_smooth_sigma}): "
                      f"{before_smooth} → {after_smooth} voxels above threshold")

            fused = fused_logits > logit_threshold
            print(f"[multi-fusion] fused occupancy ({fusion_mode}, thr={logit_threshold}): "
                  f"{fused.sum().item()} voxels from {len(conds_per_view)} views")
        else:
            # Stack bool volumes → (n_views, B, 1, reso, reso, reso)
            stacked = torch.stack(per_view_vols, dim=0)

            if fusion_mode == "union":
                fused = stacked.any(dim=0)
            elif fusion_mode == "vote":
                fused = stacked.float().mean(dim=0) >= vote_threshold
            else:
                raise ValueError(
                    f"Unknown fusion_mode: {fusion_mode!r}. "
                    f"Choose 'union', 'vote', 'logit-mean', 'logit-max', or 'logit-sum'."
                )

            print(f"[multi-fusion] fused occupancy ({fusion_mode}): {fused.sum().item()} voxels "
                  f"from {len(conds_per_view)} views")

        # Optional neighbourhood filter to remove isolated hallucination voxels
        if filter_min_neighbors > 0:
            before = int(fused.sum().item())
            fused = self._filter_isolated_voxels(fused, filter_min_neighbors)
            after = int(fused.sum().item())
            print(f"[multi-fusion] neighbour filter (min_neighbors={filter_min_neighbors}): "
                  f"{before} → {after} voxels (removed {before - after})")

        # Optionally downsample to desired resolution
        if resolution != fused.shape[2]:
            ratio = fused.shape[2] // resolution
            fused = torch.nn.functional.max_pool3d(fused.float(), ratio, ratio, 0) > 0.5

        coords = torch.argwhere(fused)[:, [0, 2, 3, 4]].int()
        return coords

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        fusion_mode: str = "union",
        vote_threshold: float = 0.5,
        logit_threshold: float = 0.0,
        logit_smooth_sigma: float = 0.0,
        samples_per_view: int = 1,
        filter_min_neighbors: int = 0,
        slat_fusion_mode: str = "primary",
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline with multiple input images fused at the sparse
        occupancy stage and optionally also at the shape SLAT feature stage.

        The sparse structure flow model is run once per image and the
        resulting occupancy volumes are fused before Shape SLAT generation.
        When ``slat_fusion_mode`` is not ``"primary"``, the shape SLAT flow
        model is also run once per image and the resulting 32-channel feature
        vectors are fused before decoding — operating at the true geometry
        representation level.

        Args:
            images (List[Image.Image]): Input images from different viewpoints.
                The first image is used as the primary conditioning image for
                Texture SLAT generation.
            num_samples (int): Number of samples (per view, typically 1).
            seed (int): Random seed.
            fusion_mode (str): ``"union"``, ``"vote"``, ``"logit-mean"``,
                ``"logit-max"``, or ``"logit-sum"``.
            vote_threshold (float): Fraction threshold for ``"vote"`` mode.
            logit_threshold (float): Threshold applied after logit fusion in
                ``logit-*`` modes (default 0.0).
            logit_smooth_sigma (float): Gaussian spatial smoothing sigma
                applied to fused logits before thresholding (0.0 = off).
            samples_per_view (int): Independent diffusion runs averaged within
                each view before cross-view fusion (1 = disabled).
            filter_min_neighbors (int): Post-fusion neighbourhood filter;
                0 = disabled.
            slat_fusion_mode (str): Shape SLAT feature fusion strategy.
                ``"primary"``          — only the first image conditions the
                                         SLAT flow model (current default).
                ``"slat-mean"``        — run SLAT flow once per view, average
                                         the 32-ch feature vectors.
                ``"slat-norm-weighted"`` — weight by per-token feature norm.
                ``"slat-max"``         — element-wise max across views.
                For cascade pipelines the LR stage always uses the primary
                view; only the HR stage is run per-view.
            sparse_structure_sampler_params (dict): Forwarded to each per-view
                sparse structure sample step.
            shape_slat_sampler_params (dict): Forwarded to shape SLAT sampling.
            tex_slat_sampler_params (dict): Forwarded to texture SLAT sampling.
            preprocess_image (bool): Apply background removal / crop to all
                images before processing.
            return_latent (bool): Whether to also return
                ``(shape_slat, tex_slat, res)``.
            pipeline_type (str): Same options as ``run()``.
            max_num_tokens (int): Token cap for cascade upsampling.

        Returns:
            List[MeshWithVoxel], or (List[MeshWithVoxel], latents) when
            ``return_latent=True``.
        """
        if not images:
            raise ValueError("At least one image is required.")

        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'tex_slat_flow_model_512' in self.models
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        elif pipeline_type in ('1024_cascade', '1536_cascade'):
            assert 'shape_slat_flow_model_512' in self.models
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        else:
            raise ValueError(f"Invalid pipeline_type: {pipeline_type}")

        if preprocess_image:
            images = [self.preprocess_image(img) for img in images]

        torch.manual_seed(seed)

        _multi_slat = slat_fusion_mode != "primary" and len(images) > 1

        # --- per-view conditioning at 512 px (used for occupancy fusion) ---
        print(f"[run_multi_image] extracting conditioning for {len(images)} image(s) ...")
        conds_512 = [self.get_cond([img], 512) for img in images]

        # --- primary conditioning for SLAT (always needed) ---
        cond_512_primary = conds_512[0]
        cond_1024_primary = self.get_cond([images[0]], 1024) if pipeline_type != '512' else None

        # --- per-view 1024-px conditioning (only needed for SLAT feature fusion) ---
        if _multi_slat and pipeline_type not in ('512',):
            print(f"[run_multi_image] extracting 1024-px conditioning for all "
                  f"{len(images)} views (slat_fusion_mode={slat_fusion_mode!r}) ...")
            conds_1024 = [self.get_cond([img], 1024) for img in images]
        else:
            conds_1024 = None

        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]

        # --- fused sparse occupancy ---
        if len(images) == 1:
            # Single-image path: reuse the existing method for consistency
            coords = self.sample_sparse_structure(
                conds_512[0], ss_res, num_samples, sparse_structure_sampler_params
            )
        else:
            coords = self.sample_sparse_structure_multi(
                conds_512, ss_res, num_samples,
                sparse_structure_sampler_params,
                fusion_mode=fusion_mode,
                vote_threshold=vote_threshold,
                logit_threshold=logit_threshold,
                logit_smooth_sigma=logit_smooth_sigma,
                samples_per_view=samples_per_view,
                filter_min_neighbors=filter_min_neighbors,
            )

        # --- shape SLAT (primary-only or per-view feature fusion) ---
        if pipeline_type == '512':
            if _multi_slat:
                shape_slat = self.sample_shape_slat_multi(
                    conds_512, self.models['shape_slat_flow_model_512'],
                    coords, shape_slat_sampler_params, slat_fusion_mode,
                )
            else:
                shape_slat = self.sample_shape_slat(
                    cond_512_primary, self.models['shape_slat_flow_model_512'],
                    coords, shape_slat_sampler_params,
                )
            tex_slat = self.sample_tex_slat(
                cond_512_primary, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params,
            )
            res = 512
        elif pipeline_type == '1024':
            if _multi_slat:
                shape_slat = self.sample_shape_slat_multi(
                    conds_1024, self.models['shape_slat_flow_model_1024'],
                    coords, shape_slat_sampler_params, slat_fusion_mode,
                )
            else:
                shape_slat = self.sample_shape_slat(
                    cond_1024_primary, self.models['shape_slat_flow_model_1024'],
                    coords, shape_slat_sampler_params,
                )
            tex_slat = self.sample_tex_slat(
                cond_1024_primary, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            if _multi_slat:
                shape_slat, res = self.sample_shape_slat_cascade_multi(
                    cond_512_primary, conds_1024,
                    self.models['shape_slat_flow_model_512'],
                    self.models['shape_slat_flow_model_1024'],
                    512, 1024,
                    coords, shape_slat_sampler_params, max_num_tokens, slat_fusion_mode,
                )
            else:
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512_primary, cond_1024_primary,
                    self.models['shape_slat_flow_model_512'],
                    self.models['shape_slat_flow_model_1024'],
                    512, 1024,
                    coords, shape_slat_sampler_params, max_num_tokens,
                )
            tex_slat = self.sample_tex_slat(
                cond_1024_primary, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
            )
        elif pipeline_type == '1536_cascade':
            if _multi_slat:
                shape_slat, res = self.sample_shape_slat_cascade_multi(
                    cond_512_primary, conds_1024,
                    self.models['shape_slat_flow_model_512'],
                    self.models['shape_slat_flow_model_1024'],
                    512, 1536,
                    coords, shape_slat_sampler_params, max_num_tokens, slat_fusion_mode,
                )
            else:
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512_primary, cond_1024_primary,
                    self.models['shape_slat_flow_model_512'],
                    self.models['shape_slat_flow_model_1024'],
                    512, 1536,
                    coords, shape_slat_sampler_params, max_num_tokens,
                )
            tex_slat = self.sample_tex_slat(
                cond_1024_primary, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
            )

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        return out_mesh

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
