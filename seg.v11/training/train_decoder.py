import argparse
import logging
import math
import os
import shutil
import json
import accelerate
import datasets
import torch
import torch.utils.data
import torch.nn.functional as F
from packaging import version
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
from decoder import maskiage_decoder 
from dataset.load import ADE20K
from utils.loss import CrossEntropyLoss
from utils.lr_scheduler import IterExponential
from utils.metric import mIoU

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Training code for maskiage decoder")
    
    # Model parameters
    parser.add_argument("--post_seg_channel", type=int, default=256)
    parser.add_argument("--post_swin_num_head", type=int, default=8) 
    parser.add_argument("--post_swin_depth", type=int, default=2)
    parser.add_argument("--post_swin_window_size", type=int, default=7)
    parser.add_argument("--num_classes", type=int, default=151)
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="decoder_checkpoints")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory",)
    parser.add_argument("--tracker_project_name", type=str, default="decoder_training", help="Project name for tracking")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint path or 'latest'")
    parser.add_argument("--lr_total_iter_length", type=int, default=20000, help="Total number of iterations for learning rate scheduling")
    parser.add_argument("--lr_exp_warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduling"
    )
    
    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    
    # Logging/saving parameters
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    # Initialize accelerator
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


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

    # Create model
    model = maskiage_decoder(
        post_seg_channel=args.post_seg_channel,
        post_swin_num_head=args.post_swin_num_head,
        post_swin_depth=args.post_swin_depth,
        post_swin_window_size=args.post_swin_window_size,
        num_classes=args.num_classes
    )


    # model loading and saving functions
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if isinstance(model, maskiage_decoder):
                        os.makedirs(os.path.join(output_dir, "decoder"), exist_ok=True)
                        torch.save(model.state_dict(), 
                                os.path.join(output_dir, "decoder", "pytorch_model.bin"))
                        config = {
                            "post_seg_channel": args.post_seg_channel,
                            "post_swin_num_head": args.post_swin_num_head,
                            "post_swin_depth": args.post_swin_depth,
                            "post_swin_window_size": args.post_swin_window_size,
                            "num_classes": args.num_classes
                        }
                        with open(os.path.join(output_dir, "decoder", "config.json"), 'w') as f:
                            json.dump(config, f)
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                if isinstance(model, maskiage_decoder):
                
                    with open(os.path.join(input_dir, "decoder", "config.json"), 'r') as f:
                        config = json.load(f)
                    
                    state_dict = torch.load(os.path.join(input_dir, "decoder", "pytorch_model.bin"))
                    model.load_state_dict(state_dict)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )

    # Learning rate scheduler
    lr_func      = IterExponential(total_iter_length = args.lr_total_iter_length*accelerator.num_processes, final_ratio = 0.01, warmup_steps = args.lr_exp_warmup_steps*accelerator.num_processes)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    # Load datasets
    train_dataset = ADE20K(root_dir="/mnt/diff_based_seg/train_dataset/ade20k/ade20k_train", transform=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )
    val_dataset = ADE20K(root_dir="/mnt/diff_based_seg/train_dataset/ade20k/ade20k_val", transform=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )

    # Prepare everything with `accelerator` (Move to GPU)
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    # Mixed precision and weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
        model.to(dtype=weight_dtype)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        model.to(dtype=weight_dtype)


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


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
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
    ce_loss = CrossEntropyLoss()

    # Training Loop
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"At Epoch {epoch}:")
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):     
                # Get inputs
                inputs = batch["seg"].to(accelerator.device)
                labels = batch["annotation"].to(accelerator.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

                estimation_loss = 0
                estimation_loss_ce = ce_loss(outputs,labels)
                if not torch.isnan(estimation_loss_ce).any():
                    estimation_loss = estimation_loss + estimation_loss_ce
                loss = loss + estimation_loss
                    
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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

                # Evaluation
                if global_step % args.eval_interval == 0:
                    eval_metrics = evaluate(args, accelerator, model, val_dataloader)
                    logger.info(f"Step {global_step} evaluation: {eval_metrics}")
                    
                # Save checkpoint
                if global_step % args.save_interval == 0:
                    save_checkpoint(args, accelerator, model, optimizer, global_step)

            # Log loss and learning rate for progress bar
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Break training
            if global_step >= args.max_train_steps:
                break
              
    # Final save
    save_checkpoint(args, accelerator, model, optimizer, global_step)
    logger.info("Training completed")
    accelerator.end_training()

def evaluate(args, accelerator, model, val_dataloader):
    """Evaluation function"""
    model.eval()
    total_miou = 0
    num_samples = 0

    val_progress = tqdm(
        enumerate(val_dataloader),  
        total=len(val_dataloader),  
        desc="Validation",
        disable=not accelerator.is_local_main_process,
        leave=False  
    )
    
    for _, batch in val_progress:
        with torch.no_grad():
            inputs = batch["seg"].to(accelerator.device)
            labels = batch["annotation"].to(accelerator.device)
            
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            mask_preds = probs.argmax(dim=1)
            
            miou_score = mIoU(mask_preds, labels)
            total_miou += miou_score
            num_samples += 1

            val_progress.set_postfix({
                    'batch_miou': f'{miou_score:.4f}',
                    'avg_miou': f'{total_miou/(num_samples):.4f}'
            })
    
    avg_miou = total_miou / num_samples
    model.train()
    
    return {"mIoU": avg_miou}

def save_checkpoint(args, accelerator, model, optimizer, step):
    """Save a checkpoint"""
    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        accelerator.save_state(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Handle checkpoints limit
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            
            # Remove old checkpoints if exceeding limit
            if len(checkpoints) > args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                removing_checkpoints = checkpoints[:num_to_remove]
                
                logger.info(f"Removing old checkpoints: {', '.join(removing_checkpoints)}")
                for ckpt in removing_checkpoints:
                    shutil.rmtree(os.path.join(args.output_dir, ckpt))

if __name__ == "__main__":
    main()