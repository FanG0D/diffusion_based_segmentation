# An official reimplemented version of Marigold training script.
# Last modified: 2024-04-29
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from diff_based_seg.diff_based_seg_pipeline import DiffBasedSegPipeline, DiffBasedSegOutput

from src.util import metric
from src.util.data_loader import skip_first_batches 
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss, CrossEntropyLoss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence


class DiffBasedSegTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: DiffBasedSegPipeline,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: DiffBasedSegPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # Adapt input layers
        if 8 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        self.model.unet.enable_xformers_memory_efficient_attention()

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(True)

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        self.optimizer = Adam(self.model.unet.parameters(), lr=lr) 

        # LR scheduler
        lr_func = IterExponential(    
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func) 

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs) 

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_seg_type = self.cfg.gt_seg_type     
        # self.gt_mask_type = self.cfg.gt_mask_type      
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

        # Get noise scheduling parameters for later conversion from a parameterized prediction into latent.
        self.alpha_prod = self.model.scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
        self.beta_prod  = 1 - self.alpha_prod


    def _replace_unet_conv_in(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3] 
        _bias = self.model.unet.conv_in.bias.clone()  # [320]

        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.5
        # new conv_in channel
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.model.unet.config["in_channels"] = 8
        logging.info("Unet config is updated")
        return

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch): 
                self.model.unet.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data
                rgb = batch["rgb_norm"].to(device)
                seg_gt_for_latent = batch[self.gt_seg_type].to(device)

                batch_size = rgb.shape[0]

                with torch.no_grad():
                    # Encode image
                    rgb_latent = self.model.encode_rgb(rgb)  # [B, 4, h, w]
                    # Encode GT SEG maskiage
                    gt_seg_latent = self.encode_seg(
                        seg_gt_for_latent
                    )  # [B, 4, h, w]

                # # Sample a random timestep for each image
                # timesteps = torch.randint(
                #     0,
                #     self.scheduler_timesteps,
                #     (batch_size,),
                #     device=device,
                #     generator=rand_num_generator,
                # ).long()  # [B]

                # Set timesteps to the first denoising step (the maximum time step)
                timesteps = torch.ones((batch_size,), device=device) * (self.scheduler_timesteps - 1)  
                timesteps = timesteps.long()  

                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        gt_seg_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        gt_seg_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    gt_seg_latent, noise, timesteps
                )  # [B, 4, h, w]

                # Text embedding
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                # Concat rgb and seg latents
                cat_latents = torch.cat(
                    [rgb_latent, noisy_latents], dim=1
                )  # [B, 8, h, w]
                cat_latents = cat_latents.float()

                # Predict the noise residual
                model_pred = self.model.unet(
                    cat_latents, timesteps, text_embed, return_dict=False
                )[0] 
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")
                

                # end to end
                alpha_prod_t = alpha_prod[timesteps].view(-1, 1, 1, 1)
                beta_prod_t  =  beta_prod[timesteps].view(-1, 1, 1, 1)

                # Convert parameterized prediction into latent prediction.
                current_latent_estimate = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * model_pred
                # Get the target for loss depending on the prediction type
                # type should be "sample"!!!

                if "sample" == self.prediction_type:
                    target = gt_seg_latent
                elif "epsilon" == self.prediction_type:
                    target = noise
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(
                        gt_seg_latent, noise, timesteps
                    )  # [B, 4, h, w]
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")

                # decode latent result
                latent_result = self.model.decode_seg(current_latent_estimate)

                # specific loss
                
                specific_loss = self.loss(latent_result.float(), target.float())
                loss = specific_loss.mean()
                self.train_metrics.update("loss", loss.item())

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dic( 
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def encode_seg(self, seg_in):
        # encode using VAE encoder
        seg_latent = self.model.encode_rgb(seg_in)
        return seg_latent

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dic[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # Save a checkpoint
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))
        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            # Read input image
            rgb_int = batch["rgb_int"]  # [1, 3, H, W]
            # GT mask-maskiage
            seg_raw_ts = batch["seg_origin"]
            seg_raw_ts = seg_raw_ts.squeeze(0)[0]
            seg_raw_ts = seg_raw_ts.to(self.device)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict seg
            pipe_out: DiffBasedSegOutput = self.model(
                rgb_int,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            seg_pred: np.ndarray = pipe_out.maskiage
            seg_pred = np.transpose(seg_pred, (1, 2, 0)) # [H, W, 3]

            mask_pred = self.rgb_to_annotation(seg_pred, self.ade_palette())
            # Evaluate
            sample_metric = []
            mask_pred_ts = torch.from_numpy(mask_pred).to(self.device) # [H, W]

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(mask_pred_ts, seg_raw_ts).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)

            # Save as 8-bit RGB png
            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")

                # Normalize maskiage from [-1, 1] to [0, 255]
                seg_to_save = ((pipe_out.maskiage + 1.0) * 127.5).astype(np.uint8)  # Map [-1, 1] to [0, 255]
                seg_to_save = np.transpose(seg_to_save, (1, 2, 0))

                # Save as an RGB image
                Image.fromarray(seg_to_save, mode="RGB").save(png_save_path)


        return metric_tracker.result()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location=self.device)
        )
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"


    def get_label_from_rgb(self, rgb, color_map):
        """
        Get the label index from RGB color using the provided color_map (palette).
        
        Parameters:
        rgb (tuple): RGB values of the pixel.
        color_map (list): The list of RGB values for each label.
        
        Returns:
        int: The label index corresponding to the RGB value.
        """
        color_map = np.array(color_map)  # Convert color_map to a numpy array
        diff = np.abs(color_map - rgb)  # Compute absolute difference between the RGB value and all color_map values
        distances = np.sum(diff, axis=1)  # Sum the differences for each color
        label = np.argmin(distances)  # The label with the smallest distance
        return label

    def rgb_to_annotation(self, seg_in_rgb, color_map):
        """
        Convert an RGB segmentation map back to an annotation map with labels.
        
        Parameters:
        seg_in_rgb (numpy array): The RGB image with shape [height, width, 3].
        color_map (list): The list of RGB values for each label.
        
        Returns:
        numpy array: The annotation map with shape [height, width] containing the label indices.
        """
        height, width, _ = seg_in_rgb.shape
        annotation_map = np.zeros((height, width), dtype=int)
        
        # Iterate through the RGB image and map each pixel to its corresponding label
        for i in range(height):
            for j in range(width):
                rgb_pixel = tuple(seg_in_rgb[i, j])  # Get RGB value for pixel (i, j)
                label = self.get_label_from_rgb(rgb_pixel, color_map)  # Get the corresponding label
                annotation_map[i, j] = label  # Set the label in the annotation map
        
        return annotation_map
    
    def ade_palette(self):
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
    