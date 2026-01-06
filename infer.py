import argparse
import logging
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.dataset.all_dataset import TestDataset
from glasswizard import GlassWizardPipeline_single_decoder_text
from src.util.seeding import seed_all


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Run single-image segmentation using GlassWizard.")
    parser.add_argument("--checkpoint", type=str, default="./weights/stable-diffusion-2", help="Checkpoint path of SD V2.",)

    # dataset setting
    parser.add_argument("--image_path", type=str, default="", help="Path to image folder.",)
    parser.add_argument("--mask_path", type=str, default="", help="Path to mask folder.")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory.")
    parser.add_argument("--fpn_path", type=str, default="./weights/fpn_decoder.pth", help="Output directory.")
    parser.add_argument("--unet_folder", type=str, default="./weights/unet", help="Output directory.")
    parser.add_argument("--embeds_repo", type=str, default='./weights/learned_embeddings_text/glass_all.safetensors')
    # inference setting
    parser.add_argument("--denoise_steps", type=int, default=1, help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",)
    parser.add_argument("--ensemble_size", type=int, default=1, help="Number of predictions to be ensembled, more inference gives better results but runs slower.",)

    # resolution setting
    parser.add_argument("--processing_res", type=int, default=0, help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.")
    parser.add_argument("--output_processing_res", action="store_true", help="When input is resized, out put depth at resized operating resolution. Default: False.",)
    parser.add_argument("--resample_method", type=str, default="bilinear", help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.")
    parser.add_argument("--input_size", type=int, default=512, help="input image size'.")

    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning("Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers.")
    resample_method = args.resample_method

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
    )

    # Random seed
    seed_all(seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    #logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    device = torch.device("cuda")

    # -------------------- Data --------------------
    test_loader = TestDataset(args.image_path, args.mask_path, args.input_size)
    
    # -------------------- Model --------------------
    dtype = torch.float32
    #dtype = torch.float16
    variant = None
    timestep_spacing = "trailing"

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderTiny
    
    unet         = UNet2DConditionModel.from_pretrained(args.unet_folder, torch_dtype=dtype)   
    vae          = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae")  
    text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder", torch_dtype=dtype)  
    tokenizer    = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer", torch_dtype=dtype) 
    scheduler    = DDIMScheduler.from_pretrained(checkpoint_path, timestep_spacing=timestep_spacing, subfolder="scheduler", torch_dtype=dtype) 
    pipe = GlassWizardPipeline_single_decoder_text.from_pretrained(pretrained_model_name_or_path = checkpoint_path,
                                                unet=unet, 
                                                vae=vae, 
                                                scheduler=scheduler, 
                                                text_encoder=text_encoder, 
                                                tokenizer=tokenizer, 
                                                variant=variant, 
                                                torch_dtype=dtype, 
                                                )
    
    embeds_repo = args.embeds_repo
    import safetensors
    saved_tensor = safetensors.torch.load_file(embeds_repo, device="cpu")
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer) + 1)
    input_embeddings = pipe.text_encoder.get_input_embeddings().weight

    # 7.4 Load token and embedding
    token = '<glass>'
    embedding = saved_tensor['<glass>']
    pipe.tokenizer.add_tokens(token)
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    input_embeddings.data[token_id] = embedding

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    logging.info(f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}")

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for i in tqdm(range(test_loader.size)):
            # Read input image
            image, _, name = test_loader.load_data()
            rgb_int = image.squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)

            # Predict depth
            pipe_out = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=0,
                color_map=None,
                show_progress_bar=False,
                resample_method=resample_method,
                fpn_path = args.fpn_path, 
            )
            
            depth_pred: np.ndarray = pipe_out.depth_np

            # Save predictions
            rgb_filename = name
            print(name)
            rgb_basename = os.path.basename(rgb_filename)
            scene_dir = os.path.join(args.output_dir, rgb_filename)
            save_to = scene_dir
            if os.path.exists(save_to):
                logging.warning(f"Existing file: '{save_to}' will be overwritten")

            pred_image = (depth_pred * 255).astype(np.uint8)
            pred_image = Image.fromarray(pred_image)
            pred_image.save(scene_dir)
