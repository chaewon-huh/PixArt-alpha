import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusion import DPMS
from diffusion.model.nets import PixArtMS_XL_2, ControlPixArtHalf
from diffusion.model.t5 import T5Embedder
from diffusers.models import AutoencoderKL
import argparse

def create_grid_image(condition_idx, image_paths, steps_values, cfg_scale_values, output_dir, base_image_size=512):
    num_steps = len(steps_values)
    num_cfg_scales = len(cfg_scale_values)

    # Assuming all images are of the same size, get size from the first image
    # If no images, return
    if not image_paths:
        print("No images provided to create_grid_image.")
        return

    # Determine individual image size (use base_image_size as fallback)
    try:
        with Image.open(image_paths[0]) as img:
            img_width, img_height = img.size
    except FileNotFoundError:
        print(f"Warning: Could not open {image_paths[0]} to determine size. Using default {base_image_size}x{base_image_size}.")
        img_width, img_height = base_image_size, base_image_size


    grid_width = num_cfg_scales * img_width
    grid_height = num_steps * img_height
    
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    print(f"\nCreating grid for condition {condition_idx}...")
    print(f"Grid dimensions: {num_cfg_scales} (cfg_scales) x {num_steps} (steps)")
    print(f"Individual image size: {img_width}x{img_height}")
    print(f"Total grid size: {grid_width}x{grid_height}")

    for i, steps in enumerate(steps_values):
        for j, cfg_scale in enumerate(cfg_scale_values):
            # Construct filename based on current convention
            # filename = f'condition{condition_idx}_steps{steps}_cfg{cfg_scale:.1f}.png'
            # More robustly find the image from the list of generated paths
            target_filename_pattern = f'condition{condition_idx}_steps{steps}_cfg{cfg_scale:.1f}.png'
            found_path = None
            for p in image_paths:
                if target_filename_pattern in p:
                    found_path = p
                    break
            
            if found_path and os.path.exists(found_path):
                try:
                    img = Image.open(found_path)
                    if img.size != (img_width, img_height):
                        img = img.resize((img_width, img_height), Image.LANCZOS) # Ensure consistent size
                    grid_image.paste(img, (j * img_width, i * img_height))
                    img.close()
                except Exception as e:
                    print(f"Error processing image {found_path}: {e}")
                    # Optionally, paste a placeholder for missing/corrupt images
                    # placeholder = Image.new('RGB', (img_width, img_height), color = 'red')
                    # grid_image.paste(placeholder, (j * img_width, i * img_height))
            else:
                print(f"Image not found for condition {condition_idx}, steps {steps}, cfg {cfg_scale:.1f} (Expected: {target_filename_pattern})")
                # Optionally, paste a placeholder for missing images
                # placeholder = Image.new('RGB', (img_width, img_height), color = 'grey')
                # grid_image.paste(placeholder, (j * img_width, i * img_height))

    grid_save_path = os.path.join(output_dir, f'condition{condition_idx}_grid_comparison.png')
    grid_image.save(grid_save_path)
    print(f"Saved grid image: {grid_save_path}")

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint', type=str, default='/workspace/PixArt-alpha/output/pixartcontrolnet-xl2-img512/checkpoints/epoch_300_step_75000.pth')
    parser.add_argument('--checkpoint', type=str, default='/workspace/PixArt-alpha/output/pixartcontrolnet-xl2-img512-fromhed/checkpoints/epoch_148_step_37000.pth')
    parser.add_argument('--condition_images', nargs='+', default=['data_val/ihchae_alpinion_04.png', 'data_val/ihchae_alpinion_05.png', 'data_val/ihchae_alpinion_06.png'])
    parser.add_argument('--prompt', type=str, default='A high-quality, photorealistic image of a healthy, anatomically normal human baby, reconstructed solely from the fetal shape and pose of the input 3D ultrasound image of an fetal.')
    parser.add_argument('--output_dir', type=str, default='test_outputs_grid')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Parameter grid for testing
    steps_values = [4, 8, 12]
    cfg_scale_values = [3.0, 4.5, 6.0]
    
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16
    vae_scale = 0.18215
    image_size = 512
    latent_size = image_size // 8
    
    print(f"Using device: {device}")
    print(f"Testing {len(steps_values)}x{len(cfg_scale_values)} parameter combinations:")
    print(f"Steps: {steps_values}")
    print(f"CFG scales: {cfg_scale_values}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    
    # VAE
    vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device).to(torch.float32)
    
    # T5
    print("Loading T5...")
    t5_model = T5Embedder(device=device, local_cache=True, cache_dir='output/pretrained_models/t5_ckpts', torch_dtype=torch.float)
    
    # PixArt ControlNet model
    print("Loading ControlNet model...")
    base_model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=1.0)
    model = ControlPixArtHalf(base_model).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    model.eval()
    model.to(weight_dtype)
    
    # Process each condition image
    for idx, condition_path in enumerate(args.condition_images):
        print(f"\nProcessing condition image {idx+1}/{len(args.condition_images)}: {condition_path}")
        
        # Load and process condition image  
        condition_img = Image.open(condition_path).convert('RGB')
        condition_img = condition_img.resize((image_size, image_size), Image.LANCZOS)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        condition_tensor = transform(condition_img).unsqueeze(0).to(device).to(torch.float32)
        
        # Directly encode the ultrasound image to latent (no HED)
        print("Encoding condition image to latent space...")
        with torch.no_grad():
            # Encode to latent
            posterior = vae.encode(condition_tensor).latent_dist
            condition_latent = posterior.sample() * vae_scale
        
        # Prepare text embedding
        print("Encoding text prompt...")
        caption_embs, emb_masks = t5_model.get_text_embeddings([args.prompt])
        caption_embs = caption_embs[:, None]
        null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]
        
        # Save original condition image once
        condition_orig_path = os.path.join(args.output_dir, f'condition{idx}_original.png')
        condition_img.save(condition_orig_path)
        print(f"Saved original: {condition_orig_path}")
        
        # Save VAE reconstruction of condition once
        condition_recon = vae.decode(condition_latent.to(torch.float32) / vae_scale).sample
        condition_recon_path = os.path.join(args.output_dir, f'condition{idx}_vae_recon.png')
        condition_recon_img = transforms.ToPILImage()(condition_recon[0].cpu().clamp(-1, 1) * 0.5 + 0.5)
        condition_recon_img.save(condition_recon_path)
        print(f"Saved VAE reconstruction: {condition_recon_path}")
        
        # Test all parameter combinations
        total_combinations = len(steps_values) * len(cfg_scale_values)
        combination_count = 0
        generated_image_paths_for_condition = [] # Store paths for current condition
        
        for steps in steps_values:
            for cfg_scale in cfg_scale_values:
                combination_count += 1
                print(f"\nTesting combination {combination_count}/{total_combinations}: steps={steps}, cfg_scale={cfg_scale}")
                
                # Set seed for reproducibility
                torch.manual_seed(args.seed)
                
                # Create noise
                z = torch.randn(1, 4, latent_size, latent_size, device=device)
                
                # Model kwargs
                hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device)
                ar = torch.tensor([1.0], device=device)
                model_kwargs = dict(
                    data_info={'img_hw': hw, 'aspect_ratio': ar}, 
                    mask=emb_masks, 
                    c=condition_latent
                )
                
                # DPM-Solver sampling
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=cfg_scale,
                    model_kwargs=model_kwargs
                )
                
                print(f"  Sampling with steps={steps}, cfg_scale={cfg_scale}...")
                with torch.no_grad():
                    samples = dpm_solver.sample(
                        z,
                        steps=steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                
                # Decode
                samples = vae.decode(samples.to(torch.float32) / vae_scale).sample
                
                # Save image with parameter info in filename
                output_path = os.path.join(args.output_dir, f'condition{idx}_steps{steps}_cfg{cfg_scale:.1f}.png')
                save_image = transforms.ToPILImage()(samples[0].cpu().clamp(-1, 1) * 0.5 + 0.5)
                save_image.save(output_path)
                print(f"  Saved: {output_path}")
                generated_image_paths_for_condition.append(output_path) # Add path to list
        
        # After processing all combinations for a condition image, create and save the grid
        if generated_image_paths_for_condition:
            create_grid_image(idx, generated_image_paths_for_condition, steps_values, cfg_scale_values, args.output_dir, base_image_size=image_size)
    
    print(f"\nAll results saved to {args.output_dir}")
    print(f"\nGrid Summary:")
    print(f"- Processed {len(args.condition_images)} condition images")
    print(f"- Tested {len(steps_values)} different steps values: {steps_values}")
    print(f"- Tested {len(cfg_scale_values)} different cfg_scale values: {cfg_scale_values}")
    print(f"- Total combinations: {len(steps_values) * len(cfg_scale_values)} per condition")
    print(f"- Total images generated: {len(args.condition_images) * len(steps_values) * len(cfg_scale_values)}")
    print(f"\nFile naming pattern: condition{{idx}}_steps{{steps}}_cfg{{cfg_scale}}.png")

if __name__ == "__main__":
    main() 