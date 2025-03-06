from typing import Dict, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torchvision.transforms.functional import resize, pil_to_tensor
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

# add
import random


def ensemble_seg(input_masks: torch.Tensor, num_classes: int):
    bsz, h, w = input_masks.shape
    votes = torch.zeros((num_classes, h, w), dtype=torch.int64, device=input_masks.device)
    for i in range(bsz):
        mask = input_masks[i]
        for c in range(num_classes):
            votes[c] += (mask == c).int()
    ensemble_mask = torch.argmax(votes, dim=0) # (h, w)

    return ensemble_mask, None


class DiffBasedSegOutput(BaseOutput):
    """
    Output class for DiffBasedSeg pipeline.
    """
    mask: np.ndarray
    maskiage: Image.Image
    uncertainty: Union[None, np.ndarray]

class DiffBasedSegPipeline(DiffusionPipeline):

    rgb_latent_scale_factor = 0.18215
    seg_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler,DDPMScheduler,LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        resample_method: str = "nearest",
        batch_size: int = 0,
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        # add
        noise = "gaussian",

    ) -> DiffBasedSegOutput:

        assert processing_res >= 0
        assert ensemble_size >= 1

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------

        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image) # [H, W, rgb] -> [rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image.squeeze()
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            3 == rgb.dim() and 3 == input_size[0]
        ), f"Wrong input shape {input_size}, expected [rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting segmentation --------------

        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # load iterator
        pred_ls  = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader

        # inference (batched)
        for batch in iterable:
            (batched_img,) = batch
            pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
            )
            pred_ls.append(pred_raw.detach())
        preds = torch.concat(pred_ls, dim=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        # remain to be fixed
        if ensemble_size > 1:   # add
            pred, pred_uncert = ensemble_seg(preds)
        else:
            pred = preds
            pred_uncert = None

        # ----------------- Post processing -----------------
        # remain to be fixed
        # if normals:
        #     # add
        #     # Normalizae normal vectors to unit length
        #     pred /= (torch.norm(pred, p=2, dim=0, keepdim=True)+1e-5)
        # else:
        #     # Scale relative prediction to [0, 1]
        #     min_d = torch.min(pred)
        #     max_d = torch.max(pred)
        #     if max_d == min_d:
        #         pred = torch.zeros_like(pred)
        #     else:
        #         pred = (pred - min_d) / (max_d - min_d)
            
        # Resize back to original resolution
        if match_input_res:
            pred = resize(
                pred.unsqueeze(0),
                (input_size[-2],input_size[-1]),
                interpolation=resample_method,
                antialias=True,
            ).squeeze()

        # Convert to numpy
        pred = pred.cpu().numpy()

        # Process prediction for visualization

        color_map = ade_palette()  
        color_map = np.array(color_map).astype(np.uint8)
        pred = pred.astype(np.uint8)
        seg_rgb = color_map[pred]
        colored_img = Image.fromarray(seg_rgb)

        return DiffBasedSegOutput(
            mask           = pred,
            maskiage       = colored_img,
            uncertainty    = pred_uncert,
        )

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
    ) -> torch.Tensor:

        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]
        
        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # add
        # Initial prediction
        latent_shape = rgb_latent.shape
        
        latent = torch.randn(
            latent_shape,
            device=device,
            dtype=self.dtype,
        )

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]

        # Denoising loop
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
                [rgb_latent, latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_step = self.scheduler.step(
                noise_pred, t, latent
            )
        
            latent = scheduler_step.prev_sample

            # add
            if i == num_inference_steps-1:
                latent = scheduler_step.pred_original_sample 
        
        # decode maskiage and transform to mask
        maskiage = self.decode_maskiage(latent)
        maskiage = (maskiage + 1.0)/ 2.0
        mask = self.decode_from_segmap(maskiage * 255, keep_ignore_index=True, prob = False).squeeze(1)
        return mask


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent
       
    def decode_maskiage(self, maskiage_latent: torch.Tensor) -> torch.Tensor:
        # scale latent
        maskiage_latent = maskiage_latent / self.seg_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(maskiage_latent)
        maskiage = self.vae.decoder(z)
        return maskiage
    
    def decode_from_segmap(self, segmap, keep_ignore_index, prob=False):
        colormap = torch.tensor(ade_palette())
        PALETTE_ = colormap.clone().to(segmap.device) \
            if keep_ignore_index \
            else colormap[:-1].clone().to(segmap.device) # N, C

        B, C, H, W = segmap.shape # B, N, C, H, W
        N, _ = PALETTE_.shape
        p = PALETTE_.reshape(1, N, C, 1, 1)
        if keep_ignore_index:
            segmap = torch.Tensor.repeat(segmap, 150 + 1, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        else:
            segmap = segmap.reshape(B, 1, C, H, W)
        if prob:
            return ((segmap - p) ** 2).sum(2)
        else:
            return ((segmap - p) ** 2).sum(2).argmin(1).unsqueeze(1)
        
def ade_palette():
    """ADE20K palette for external use."""
    return  [[222, 222, 145], [18, 30, 7], [8, 23, 47], [30, 6, 96],
                [1, 13, 164], [12, 28, 191], [25, 52, 32], [29, 48, 52],
                [15, 51, 95], [25, 56, 167], [25, 42, 210], [27, 81, 31],
                [9, 88, 54], [27, 92, 113], [11, 99, 151], [26, 110, 183],
                [24, 130, 26], [4, 122, 75], [3, 132, 98], [26, 147, 167],
                [17, 132, 197], [5, 169, 28], [19, 184, 67], [0, 190, 122],
                [12, 167, 147], [6, 161, 196], [2, 205, 3], [5, 220, 61],
                [23, 225, 107], [7, 217, 157], [25, 208, 191], [74, 10, 8],
                [69, 30, 69], [56, 4, 98], [61, 29, 164], [60, 10, 194],
                [60, 52, 19], [74, 69, 52], [65, 68, 116], [81, 41, 161],
                [70, 60, 197], [66, 81, 14], [55, 107, 61], [76, 110, 108],
                [74, 104, 162], [72, 94, 197], [60, 133, 16], [69, 128, 67],
                [59, 148, 104], [65, 133, 154], [68, 128, 183], [79, 181, 11],
                [76, 170, 56], [71, 175, 103], [53, 162, 137], [53, 182, 183],
                [51, 229, 26], [51, 202, 51], [69, 213, 122], [63, 213, 161],
                [71, 203, 197], [120, 11, 31], [124, 3, 68], [131, 2, 98],
                [113, 1, 162], [102, 13, 209], [109, 50, 30], [126, 41, 47],
                [107, 46, 118], [112, 49, 147], [109, 41, 189], [103, 83, 15],
                [126, 99, 70], [124, 101, 104], [131, 103, 159],
                [128, 110, 183], [119, 148, 9], [112, 137, 50], [123, 127, 116],
                [107, 124, 167], [102, 148, 203], [124, 180, 15],
                [116, 168, 65], [104, 182, 102], [111, 164, 163],
                [105, 174, 191], [102, 218, 20], [126, 203, 64],
                [108, 215, 109], [110, 221, 157], [107, 230, 192],
                [160, 25, 11], [165, 12, 65], [153, 2, 117], [182, 21, 141],
                [160, 19, 188], [176, 58, 19], [175, 58, 56], [170, 69, 93],
                [176, 42, 146], [157, 44, 211], [157, 105, 2], [180, 98, 73],
                [182, 85, 92], [169, 93, 152], [156, 89, 202], [157, 144, 22],
                [180, 151, 77], [154, 146, 118], [162, 136, 143],
                [171, 134, 184], [170, 174, 15], [178, 180, 65],
                [176, 183, 120], [175, 169, 147], [181, 165, 197],
                [156, 227, 3], [167, 218, 61], [160, 216, 119],
                [164, 251, 141], [177, 201, 251], [231, 30, 13], [219, 6, 59],
                [211, 26, 122], [216, 16, 153], [209, 12, 192], [216, 70, 15],
                [215, 46, 60], [234, 61, 112], [224, 53, 157], [227, 49, 207],
                [221, 108, 8], [220, 93, 73], [230, 111, 113], [218, 89, 143],
                [231, 90, 195], [227, 144, 22], [208, 137, 49], [210, 128, 116],
                [225, 135, 157], [221, 135, 193], [211, 174, 18],
                [222, 185, 50], [229, 183, 93], [233, 162, 155],
                [255, 167, 205], [211, 215, 15], [232, 225, 71], [0, 0, 0],
                [255, 255, 255], [215, 216, 196]]