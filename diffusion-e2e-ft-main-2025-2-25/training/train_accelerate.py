import argparse
import logging
import math
import os
import shutil

import accelerate
import datasets
import torch
import torch.utils.checkpoint
import transformers
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from torch.optim.lr_scheduler import LambdaLR
# remain to be fixed
from dataloaders.load import *
from util.noise import pyramid_noise_like
from util.loss import CrossEntropyLoss
from util.unet_prep import replace_unet_conv_in
from util.lr_scheduler import IterExponential
from util.metric import mIoU

if is_wandb_available():
    import wandb

check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Training code for 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think'.")
    # Our settings:
    parser.add_argument(
        "--modality",
        type=str,
        choices=["seg"],
        required=True,
    )
    parser.add_argument(
        "--noise_type", 
        type=str,
        default=None, # If left as None, Stable Diffusion checkpoints can be trained without altering the input channels (i.e., only 4 input channels for the RGB input).
        choices=["zeros", "gaussian", "pyramid"],
        help="If left as None, Stable Diffusion checkpoints can be trained without altering the input channels (i.e., only 4 input channels for RGB input)."
    )
    parser.add_argument(
        "--lr_exp_warmup_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr_total_iter_length",
        type=int,
        default=20000,
    )
    # Stable diffusion training settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/model-finetuned",
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=2, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=15,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        required=True,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="e2e-ft-diffusion",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # Validation arguments
    parser.add_argument(
        "--val_batch_size",
        default=1,
        type=int,
        help="Batch size (per device) for validation"
    )
    parser.add_argument(
        "--start_eval_epoch",
        default=0,
        type=int,
        help="Start validation from this epoch"
    ) 
    parser.add_argument(
        "--eval_interval",
        default=1000,
        type=int,
        help="Run validation every n steps"
    )
    parser.add_argument(
        "--vis_interval",
        default=500,
        type=int,
        help="Run validation every n steps"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def encode_image(vae, image):
    h = vae.encoder(image)
    moments = vae.quant_conv(h)
    latent, _ = torch.chunk(moments, 2, dim=1)
    return latent

def decode_image(vae, latent):
    z = vae.post_quant_conv(latent)
    image = vae.decoder(z)
    return image

def decode_from_segmap(segmap, keep_ignore_index, prob=False):
    colormap = torch.tensor(ade_palette())
    PALETTE_ = colormap.clone().to(segmap.device) \
        if keep_ignore_index \
        else colormap[:-1].clone().to(segmap.device) # N, C

    # print(f"Input segmap shape: {segmap.shape}")
    # print(f"PALETTE_ shape: {PALETTE_.shape}")

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

def main():
    args = parse_args()

    # Init accelerator and logger
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # remain to be fixed
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Save training arguments in a .txt file
    if accelerator.is_main_process:
        args_dict = vars(args)
        args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
        args_path = os.path.join(args.output_dir, "arguments.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args_path, 'w') as file:
            file.write(args_str)
    if args.noise_type is None:
        logger.warning("Noise type is `None`. This setting is only meant for checkpoints without image conditioning, such as Stable Diffusion.")

    # Load model components
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    unet         = UNet2DConditionModel.from_pretrained( args.pretrained_model_name_or_path, subfolder="unet", revision=None)

    if args.noise_type is not None:
        # Double UNet input layers if necessary
        if unet.config['in_channels'] != 8:
            replace_unet_conv_in(unet, repeat=2)
            logger.info("Unet conv_in layer is replaced for RGB-depth or RGB-normals input")

    # Freeze or set model components to training mode
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    # logits_decoder.train()

    # Use xformers for efficient attention
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Diffusers model loading and saving functions
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if isinstance(model, UNet2DConditionModel):
                        os.makedirs(os.path.join(output_dir, "unet"), exist_ok=True)
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    # elif isinstance(model, torch.nn.Conv2d):
                    #     os.makedirs(os.path.join(output_dir, "logits_decoder"), exist_ok=True)
                    #     torch.save(model.state_dict(), os.path.join(output_dir, "logits_decoder", "pytorch_model.bin"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                if isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                # elif isinstance(model, torch.nn.Conv2d):
                #     state_dict = torch.load(os.path.join(input_dir, "logits_decoder", "pytorch_model.bin"))
                #     model.load_state_dict(state_dict)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),        #list(unet.parameters()) + list(logits_decoder.parameters()), 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Learning rate scheduler
    lr_func      = IterExponential(total_iter_length = args.lr_total_iter_length*accelerator.num_processes, final_ratio = 0.01, warmup_steps = args.lr_exp_warmup_steps*accelerator.num_processes)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    # Training datasets
    ade20k_train_root_dir = "/root/autodl-tmp/train_dataset/ade20k/ade20k_train"
    train_dataset_ade20k = ADE20K(root_dir=ade20k_train_root_dir, transform=True)
    train_dataloader_ade20k = torch.utils.data.DataLoader(train_dataset_ade20k, shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)

    # Validation datasets
    ade20k_val_root_dir = "/root/autodl-tmp/train_dataset/ade20k/ade20k_val"
    val_dataset_ade20k = ADE20K(root_dir=ade20k_val_root_dir, transform=False)
    val_dataloader_ade20k = torch.utils.data.DataLoader(val_dataset_ade20k, shuffle=False, batch_size=args.val_batch_size, num_workers=args.dataloader_num_workers)

    # Visualization datasets
    ade20k_vis_root_dir = "/root/autodl-tmp/train_dataset/ade20k/ade20k_vis"
    vis_dataset_ade20k = ADE20K(root_dir=ade20k_vis_root_dir, transform=False)
    vis_dataloader_ade20k = torch.utils.data.DataLoader(vis_dataset_ade20k, shuffle=False, batch_size=args.val_batch_size, num_workers=args.dataloader_num_workers)

    # Prepare everything with `accelerator` (Move to GPU)
    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader_ade20k, val_dataloader_ade20k, lr_scheduler)

    # Mixed precision and weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
        # unet.to(dtype=weight_dtype)
        # logits_decoder.to(dtype=weight_dtype)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        # unet.to(dtype=weight_dtype)
        # logits_decoder.to(dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)    
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Calculate number of training steps and epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_ade20k)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Resume training from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:  
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps), 
        initial=initial_global_step, 
        desc="Steps", 
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,) 

    # Init task specific losses
    mse_loss = torch.nn.MSELoss(reduction='mean')


    # Pre-compute empty text CLIP encoding
    empty_token    = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
    empty_token    = empty_token.to(accelerator.device)
    empty_encoding = text_encoder(empty_token, return_dict=False)[0]
    empty_encoding = empty_encoding.to(accelerator.device)

    # Training Loop
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"At Epoch {epoch}:")
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):     
                
                # RGB latent
                rgb_latents = encode_image(vae, batch["rgb"].to(device=accelerator.device, dtype=weight_dtype))
                rgb_latents = rgb_latents * vae.config.scaling_factor
                rgb_latents_repeated = rgb_latents.repeat(2, 1, 1, 1) # [2*B, C, H, W]
                
                # seg latent
                seg_latents_origin = encode_image(vae, batch["seg"].to(device=accelerator.device, dtype=weight_dtype))
                seg_latents = seg_latents_origin * vae.config.scaling_factor

                # Set timesteps to the first denoising step
                timesteps = torch.ones((rgb_latents.shape[0],), device=rgb_latents.device) * (noise_scheduler.config.num_train_timesteps-1) # 999
                # timesteps = timesteps.repeat(2) # [2*B]
                timesteps = timesteps.long()
                
                task_class = torch.tensor([[0, 1], [1, 0]], device=accelerator.device, dtype=weight_dtype)
                task_emb = torch.cat([torch.sin(task_class), torch.cos(task_class)], dim=-1) # [2, 4]
                task_emb = task_emb.repeat_interleave(rgb_latents.shape[0], 0) # [2*B, 4]

                # Sample noisy latent
                if (args.noise_type is None) or (args.noise_type == "zeros"):
                    noisy_latents = torch.zeros_like(rgb_latents).to(accelerator.device)
                elif args.noise_type == "pyramid":
                    noisy_latents = pyramid_noise_like(rgb_latents).to(accelerator.device)
                elif args.noise_type == "gaussian":
                    noise = torch.randn_like(seg_latents).to(accelerator.device)
                    noisy_latents = noise_scheduler.add_noise(seg_latents, noise, timesteps)
                    # noisy_latents = torch.randn_like(rgb_latents).to(accelerator.device)
                else:
                    raise ValueError(f"Unknown noise type {args.noise_type}")

                target = torch.cat((rgb_latents, seg_latents), dim=0)

                # Generate UNet prediction
                encoder_hidden_states = empty_encoding.repeat(len(batch["rgb"]), 1, 1)
                encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
                unet_input = (
                    torch.cat((rgb_latents_repeated, torch.cat((rgb_latents, noisy_latents), dim=0)), dim=1).to(accelerator.device)
                    if args.noise_type is not None
                    else rgb_latents
                )

                model_pred = unet(unet_input, timesteps.repeat(2), encoder_hidden_states, return_dict=False, class_labels=task_emb,)[0]

                # End-to-end fine-tuning 
                loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                estimation_loss = 0
                # recon_pred, seg_pred = torch.chunk(model_pred, 2, dim=0)
                # recon_target, seg_target = torch.chunk(target, 2, dim=0)
                # rec_mse_loss = mse_loss(recon_pred, recon_target)
                # seg_mse_loss = mse_loss(seg_pred, seg_target)
                total_mse_loss = mse_loss(model_pred.float(), target.float())

                # if not torch.isnan(rec_mse_loss).any():
                #     estimation_loss = estimation_loss + rec_mse_loss

                # if not torch.isnan(seg_mse_loss).any():
                #     estimation_loss = estimation_loss + seg_mse_loss


                if not torch.isnan(total_mse_loss).any():
                    estimation_loss = estimation_loss + total_mse_loss
                loss = loss + estimation_loss
                    
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                # Validation & Visualization
                if global_step % args.eval_interval == 0:
                    miou = validation_step(
                        args,
                        accelerator, 
                        (unet, noise_scheduler, vae, tokenizer, text_encoder),
                        val_dataloader_ade20k
                    )
                    
                    accelerator.log({
                        "val_miou": miou,
                        "step": global_step,
                        "epoch": epoch,
                    }, step=global_step)   

                if global_step % args.vis_interval == 0:    
                    # Visualization
                    if accelerator.is_main_process:
                        visualization_step(
                            args,
                            accelerator, 
                            (unet, noise_scheduler, vae, tokenizer, text_encoder),
                            vis_dataloader_ade20k,
                            global_step
                        )
                        
                # Save model checkpoint 
                if global_step % args.checkpointing_steps == 0:
                    logger.info(f"Entered Saving Code at global step {global_step} checkpointing_steps {args.checkpointing_steps}")
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            # Log loss and learning rate for progress bar
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Break training
            if global_step >= args.max_train_steps:
                break
              
    # Create SD pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler", 
            timestep_spacing="trailing", # set scheduler timestep spacing to trailing for later inference.
            revision=args.revision, 
            variant=args.variant
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            revision=args.revision,
            variant=args.variant,
        )
        logger.info(f"Saving pipeline to {args.output_dir}")
        pipeline.save_pretrained(args.output_dir)
        
        # # Save logits_decoder
        # logits_decoder_path = os.path.join(args.output_dir, "logits_decoder")
        # os.makedirs(logits_decoder_path, exist_ok=True)
        # torch.save(logits_decoder.state_dict(), os.path.join(logits_decoder_path, "pytorch_model.bin"))
    logger.info(f"Finished training.")

    accelerator.end_training()

def validation_step(args, accelerator, model_components, val_dataloader):
    """Validation step during training"""
    logger.info("Running validation...")
    unet, noise_scheduler, vae, tokenizer, text_encoder = model_components
    # weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    # Switch to eval mode
    unet.eval() 
    # logits_decoder.eval()
    
    # Initialize metrics
    total_miou = 0.0
    num_samples = 0

    val_progress = tqdm(
        enumerate(val_dataloader),  
        total=len(val_dataloader),  
        desc="Validation",
        disable=not accelerator.is_local_main_process,
        leave=False  
    )

    # Get validation batch size from dataloader
    val_batch_size = val_dataloader.batch_size
    assert 1 == val_batch_size

    for step, batch in val_progress:
        with torch.no_grad():
            # Get RGB latents
            rgb_latents = encode_image(vae, batch["rgb"].to(device=accelerator.device))
            rgb_latents = rgb_latents * vae.config.scaling_factor
            
            # Get ground truth segmentation
            ground_truth = batch["annotation"].to(device=accelerator.device)
            
            # noise
            latent = torch.randn(
                rgb_latents.shape,
                device=accelerator.device,
                dtype=rgb_latents.dtype,
            )

            # Get empty text encoding
            empty_token    = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
            empty_token    = empty_token.to(accelerator.device)
            empty_encoding = text_encoder(empty_token, return_dict=False)[0]
            empty_encoding = empty_encoding.to(accelerator.device)

            # Set denoising step = 1 
            noise_scheduler.set_timesteps(1, device=accelerator.device)
            timesteps = noise_scheduler.timesteps

            # Generate UNet prediction 
            encoder_hidden_states = empty_encoding.repeat(len(batch["rgb"]), 1, 1)
            unet_input = torch.cat((rgb_latents, latent), dim=1)
            task_emb = torch.tensor(
                [1, 0], 
                device=accelerator.device, 
            ).unsqueeze(0).repeat(1, 1)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
            x0_pred = unet(unet_input, timesteps, encoder_hidden_states)[0]

            # # compute the previous noisy sample x_t -> x_t-1
            # scheduler_step = noise_scheduler.step(
            #     noise_pred, timesteps, latent
            # )
            
            # latent = scheduler_step.pred_original_sample

            latent = x0_pred

            # Decode latent and get segmentation prediction
            latent = latent / vae.config.scaling_factor
            current_estimate = decode_image(vae, latent)
            current_estimate = (current_estimate + 1.0) / 2.0
            mask = decode_from_segmap(current_estimate * 255, keep_ignore_index=True, prob=False).squeeze(1)
            # logits_pred = logits_decoder(current_estimate)
            # probs = torch.softmax(logits_pred, dim=1)
            # mask = torch.argmax(probs, dim=1)

            # Calculate metrics
            batch_miou = mIoU(mask, ground_truth)
            total_miou += batch_miou
            num_samples += 1
            
            del rgb_latents, latent, x0_pred
            torch.cuda.empty_cache()
            
            val_progress.set_postfix({
                    'batch_miou': f'{batch_miou:.4f}',
                    'avg_miou': f'{total_miou/(num_samples):.4f}'
            })

    avg_miou = total_miou / num_samples

    if accelerator.is_local_main_process:
        logger.info(f"Validation completed. Average mIoU: {avg_miou:.4f}")

    # Switch back to train mode
    unet.train()
    # logits_decoder.train()
    
    return avg_miou

def visualization_step(args, accelerator, model_components, vis_dataloader, step):
    """Visualization step during training"""
    logger.info("Running visualization...")
    unet, noise_scheduler, vae, tokenizer, text_encoder = model_components
    # weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    # Switch to eval mode
    unet.eval() 
    # logits_decoder.eval()
    
    vis_dir = os.path.join(args.output_dir, f"visualization/step_{step}")
    if accelerator.is_main_process:
        os.makedirs(vis_dir, exist_ok=True)

    # Get visualization batch size from dataloader
    vis_batch_size = vis_dataloader.batch_size
    assert 1 == vis_batch_size

    for idx, batch in enumerate(vis_dataloader):
        with torch.no_grad():
            # Get RGB latents
            rgb_latents = encode_image(vae, batch["rgb"].to(device=accelerator.device))
            rgb_latents = rgb_latents * vae.config.scaling_factor
            
            # noise
            latent = torch.randn(
                rgb_latents.shape,
                device=accelerator.device,
                dtype=rgb_latents.dtype,
            )

            # Get empty text encoding
            empty_token    = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
            empty_token    = empty_token.to(accelerator.device)
            empty_encoding = text_encoder(empty_token, return_dict=False)[0]
            empty_encoding = empty_encoding.to(accelerator.device)

            # Set denoising step = 1 
            noise_scheduler.set_timesteps(1, device=accelerator.device)
            timesteps = noise_scheduler.timesteps

            # Generate UNet prediction 
            encoder_hidden_states = empty_encoding.repeat(len(batch["rgb"]), 1, 1)
            unet_input = torch.cat((rgb_latents, latent), dim=1)
            task_emb = torch.tensor(
                [1, 0], 
                device=accelerator.device
            ).unsqueeze(0).repeat(1, 1)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
            x0_pred = unet(unet_input, timesteps, encoder_hidden_states)[0]
            latent = x0_pred

            # Decode latent and get segmentation prediction
            latent = latent / vae.config.scaling_factor
            current_estimate = decode_image(vae, latent)
            current_estimate = (current_estimate + 1.0) / 2.0
            mask = decode_from_segmap(current_estimate * 255, keep_ignore_index=True, prob=False).squeeze().cpu().numpy()

            if accelerator.is_main_process:
                
                img_name = f"vis_{idx+1}"
                png_save_path = os.path.join(vis_dir, f"{img_name}.png")

                color_map = ade_palette()  
                color_map = np.array(color_map).astype(np.uint8)
                mask = mask.astype(np.uint8)
                seg_to_save = color_map[mask]

                Image.fromarray(seg_to_save, mode="RGB").save(png_save_path)
            
            del rgb_latents, latent, x0_pred
            torch.cuda.empty_cache()

    
    # Switch back to train mode
    unet.train()
    # logits_decoder.train()

    logger.info(f"Visualization completed. Results saved to {vis_dir}")

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

if __name__ == "__main__":
    main()