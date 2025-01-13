import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_seg
from .util.image_util import (
    get_tv_resample_method,
    resize_max_res,
)

class DiffBasedSegOutput(BaseOutput):
    """
    Output class for DiffBasedSeg pipeline.
    """
    maskiage: np.ndarray
    uncertainty: Union[None, np.ndarray]

class DiffBasedSegPipeline(DiffusionPipeline):
    """
    DiffBasedSeg pipeline.
    """
    rgb_latent_scale_factor = 0.18215
    seg_latent_scale_factor = 0.18215

    def __init__(
            self, 
            vae: AutoencoderKL, 
            unet: UNet2DConditionModel, 
            scheduler: Union[DDIMScheduler, LCMScheduler], text_encoder: CLIPTextModel, 
            tokenizer: CLIPTokenizer, 
            default_denoising_steps: Optional[int] = None, 
            default_processing_resolution: Optional[int] = None
        ):
            
        super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,

    ) -> DiffBasedSegOutput:

        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution
        
        assert processing_res >= 0
        assert ensemble_size >= 1

        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        
        input_size = rgb.shape
        assert 4 == rgb.dim() and 3 == input_size[-3], f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb, 
                max_edge_resolution=processing_res, resample_method=resample_method
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # --------------predicting maskiage---------------------
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)# [B, 3, H, W]
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(ensemble_size=ensemble_size, input_res=max(rgb.shape[1:]), dtype=self.dtype)
        
        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)

        # predict maskiage
        maskiage_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False)
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            maskiage_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            maskiage_pred_ls.append(maskiage_pred_raw.detach())
        maskiage_preds = torch.concat(maskiage_pred_ls, dim=0) # [ensemble_size, 3, H, W]
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            maskiage_pred, pred_uncert = ensemble_seg(
                maskiage_preds,
                max_res=50,
                **(ensemble_kwargs or {}),
            )
        else:
            maskiage_pred = maskiage_preds
            pred_uncert = None

        # Resize back to original resolution
        if match_input_res:
            maskiage_pred = resize(
                maskiage_pred,  # [1, 3, H, W]
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        maskiage_pred = maskiage_pred.squeeze()
        maskiage_pred = maskiage_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()


        return DiffBasedSegOutput(
            maskiage=maskiage_pred,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)


    @torch.no_grad()
    def single_infer(
    self,
    rgb_in: torch.Tensor, 
    num_inference_steps: int, 
    show_pbar: bool, 
    generator: Union[torch.Generator, None],
    ) -> torch.Tensor:
        
        device = self.device
        rgb_in = rgb_in.to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        rgb_latent = self.encode_rgb(rgb_in)

        maskiage_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )# [B, 4, h, w]

        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)

        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        
        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, maskiage_latent], dim=1
            )

            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample # [B, 4, h, w]
            
            # compute the previous noisy sample x_t -> x_t-1
            maskiage_latent = self.scheduler.step(
                noise_pred, t, maskiage_latent, generator=generator
            ).prev_sample

        maskiage = self.decode_seg(maskiage_latent)
        # #  [0, 255] -> [-1, 1]
        # maskiage = maskiage / 255.0 * 2.0 - 1.0  
        return maskiage

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode segmentation map into latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent # [B, 4, h, w]
    
    def decode_seg(self, seg_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode segmentation latent into segmentation map.
        """
        # scale latent
        seg_latent = seg_latent / self.seg_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(seg_latent)
        maskiage = self.vae.decoder(z)
        return maskiage


