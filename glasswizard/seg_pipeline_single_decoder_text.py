# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
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
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
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
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    def __init__(self, in_channels=4, feature_channels=64, num_out_channels=1):
        super(FPNDecoder, self).__init__()
        
        self.latent_conv = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        
        self.output_conv = nn.Conv2d(feature_channels, num_out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):

        p = self.latent_conv(x)
        p3 = F.relu(p)  # [B, feature_channels, h, w]
        p2 = F.interpolate(p3, scale_factor=2, mode='nearest')  # 上采样2倍
        p2 = F.relu(self.conv2(p2))  # [B, feature_channels, h*2, w*2]
        p1 = F.interpolate(p2, scale_factor=4, mode='nearest')  # 再上采样2倍
        p1 = F.relu(self.conv1(p1))  # [B, feature_channels, h*4, w*4]

        output = self.output_conv(p1)  # [B, num_out_channels, h*4, w*4]

        return output


class GlassWizardOutput(BaseOutput):
    """

    Args:
        depth_np (`np.ndarray`): Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`): Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`): Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]

class GlassWizardPipeline_single_decoder_text(DiffusionPipeline):

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, DDPMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None
        
        # ========Decoder
        self.fpn_decoder = FPNDecoder()
        
        
    def load_fpn_decoder_weights(self, weights_path: str):

        state_dict = torch.load(weights_path, map_location=self.device)
        self.fpn_decoder.load_state_dict(state_dict)
        self.fpn_decoder.to(self.device)
        self.fpn_decoder.eval()
        

    @torch.no_grad()
    def __call__(self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        fpn_path: str = None,
    ) -> GlassWizardOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`): Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        
        self.load_fpn_decoder_weights(fpn_path)
        self.fpn_decoder.cuda()
        # Model-specific optimal default values leading to fast and reasonable results.
        assert processing_res >= 0
        assert ensemble_size >= 1

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")# convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)                  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (4 == rgb.dim() and 3 == input_size[-3]), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(rgb, max_edge_resolution=processing_res, resample_method=resample_method,)

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(ensemble_size=ensemble_size, input_res=max(rgb_norm.shape[1:]), dtype=self.dtype,)

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)

        # Predict depth maps (batched)
        seg_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False)
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            seg_pred_raw = self.single_infer(rgb_in=batched_img.to(self.dtype),
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar)
            seg_pred_ls.append(seg_pred_raw.detach())
        seg_preds = torch.concat(seg_pred_ls, dim=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        seg_pred = seg_preds

        # Resize back to original resolution
        if match_input_res:
            seg_pred = resize(seg_pred, input_size[-2:], interpolation=resample_method, antialias=True,)

        # Convert to numpy
        seg_pred = seg_pred.cpu().numpy()

        # Colorize
        if color_map is not None:
            depth_colored = colorize_depth_maps(seg_pred, 0, 1, cmap=color_map).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return GlassWizardOutput(
            depth_np=seg_pred,
            depth_colored=depth_colored_img,
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
                logging.warning(f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference.")
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps.")
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
        
    def encode_text(self, prompt):
        """
        Encode text embedding for empty prompt
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
        

    @torch.no_grad()
    def single_infer(self, 
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        show_pbar: bool,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`): Input RGB image.
            num_inference_steps (`int`): Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`): Display a progress bar of diffusion denoising.
            generator (`torch.Generator`) Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        
        rgb_latent = self.encode_rgb(rgb_in)

        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
                
        timesteps = torch.tensor([999], device=device).long()
        self.scheduler.set_timesteps(1, device=device)

        noise = torch.zeros(rgb_latent.shape, device=device)
        
        # Text embedding
        #if self.empty_text_embed is None:
        #     self.encode_empty_text()
        #text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1)).to(device)  # [B, 2, 1024]
        
        #caption = torch.tensor([[49406,   320,  1125,   539,  3313, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device).repeat((rgb_latent.shape[0], 1))
        #caption_enc = self.text_encoder(caption)[0]
        self.encode_text('A photo with <glass>')
        #self.encode_text('')
        #self.encode_text('A photo with glass')
        #self.encode_text('A photo with transparent objects')
            
        #caption_enc = self.encode_text('A photo with <glass>').repeat((rgb_latent.shape[0], 1, 1))
        caption_enc = self.text_embed.to(device).repeat((rgb_latent.shape[0], 1, 1))
        #text_embed = self.empty_text_embed.to(device).repeat((rgb_latent.shape[0], 1, 1))  # [B, 77, 1024]
        # Concat rgb and depth latents
        cat_latents = torch.cat([rgb_latent, noise], dim=1)  # [B, 8, h, w]
        cat_latents = cat_latents.to(self.dtype)
        # Predict the noise residual
        model_pred = self.unet(cat_latents, timesteps, caption_enc).sample  # [B, 4, h, w]
        step_pred = self.scheduler.step(model_pred, timesteps, noise, return_dict=True).prev_sample

        seg_pred = self.fpn_decoder(step_pred)
        
        seg = torch.sigmoid(seg_pred)
        

        return seg

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:

        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:

        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
    