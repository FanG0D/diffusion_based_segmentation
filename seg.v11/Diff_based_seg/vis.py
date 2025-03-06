import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sys
from diff_based_seg import DiffBasedSegPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
sys.path.append(os.getcwd())

from src.util.seeding import seed_all

# remain to be fixed
from src.dataset import (
    BaseSegDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run semantic segmentation using DiffBasedSeg."
    )
    # remain to be fixed
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="diff_based_seg_lotus",
        help="Checkpoint path or hub name.",
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        default="nearest",
        help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.",
    )

    # add
    parser.add_argument(
        "--noise",
        type=str,
        default='gaussian',
        choices=["gaussian", "pyramid", "zeros"],
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default='trailing',
        choices=["trailing", "leading"],
    )


    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    # add
    noise = args.noise
    timestep_spacing = args.timestep_spacing

    checkpoint_path = args.checkpoint
    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed

    # add
    # Save arguments in txt file
    print(f"arguments: {args}")
    parent_output_dir = os.path.dirname(args.output_dir)
    os.makedirs(parent_output_dir, exist_ok=True)
    args_dict = vars(args)
    args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
    args_path = os.path.join(parent_output_dir, "arguments.txt")
    with open(args_path, 'w') as file:
        file.write(args_str)
    print(f"Arguments saved in {args_path}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseSegDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.RGB_ONLY
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    # add
    print(f"Loading model from {checkpoint_path}")
    unet         = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")   
    vae          = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae")  
    text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")  
    tokenizer    = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer") 
    scheduler    = DDIMScheduler.from_pretrained(checkpoint_path, timestep_spacing=timestep_spacing, subfolder="scheduler") 
    pipe = DiffBasedSegPipeline.from_pretrained(pretrained_model_name_or_path = checkpoint_path,
                                            unet=unet, 
                                            vae=vae, 
                                            scheduler=scheduler, 
                                            text_encoder=text_encoder, 
                                            tokenizer=tokenizer, 
                                            variant=variant, 
                                            torch_dtype=dtype)
        
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    pipe.unet.eval()

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True
        ):
            # Read input image
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)

            pipe_out = pipe(
                    input_image,
                    denoising_steps=denoise_steps,
                    ensemble_size=ensemble_size,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    batch_size=0,
                    show_progress_bar=False,
                    resample_method=resample_method,
                )

            mask_pred: np.ndarray = pipe_out.mask
            maskiage_pred: Image.Image =pipe_out.maskiage

            # Save predictions
            rgb_filename = batch["rgb_relative_path"][0]
            rgb_basename = os.path.basename(rgb_filename)

            maskiage_basename = get_pred_name(rgb_basename, dataset.name_mode, suffix=".png")

            maskiage_save_to = os.path.join(output_dir, maskiage_basename)
            if os.path.exists(maskiage_save_to):
                logging.warning(f"Existing file: '{maskiage_save_to}' will be overwritten")
            maskiage_pred.save(maskiage_save_to)