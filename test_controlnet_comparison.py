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
import glob

def create_simple_comparison(condition_img, generated_img):
    """Create simple side-by-side comparison image without labels"""
    # Resize images to same size if needed
    width, height = condition_img.size
    if generated_img.size != (width, height):
        generated_img = generated_img.resize((width, height), Image.LANCZOS)
    
    # Create comparison canvas (just concatenate horizontally)
    comparison_width = width * 2
    comparison_height = height
    comparison_img = Image.new('RGB', (comparison_width, comparison_height))
    
    # Paste images side by side
    comparison_img.paste(condition_img, (0, 0))
    comparison_img.paste(generated_img, (width, 0))
    
    return comparison_img

def get_image_files(directory):
    """Get all image files from directory"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    # Sort files for consistent ordering
    image_files.sort()
    return image_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/workspace/PixArt-alpha/output/pixartcontrolnet-xl2-img512/checkpoints/epoch_320_step_80000.pth')
    parser.add_argument('--input_dir', type=str, default='data_val', help='Directory containing input images')
    parser.add_argument('--prompt', type=str, default='A high-quality, photorealistic image of a healthy, anatomically normal human baby, reconstructed solely from the fetal shape and pose of the input 3D ultrasound image of an fetal.')
    parser.add_argument('--output_dir', type=str, default='test_outputs_comparison')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Fixed parameters for comparison
    steps = 20
    cfg_scale = 6.0
    
    # Get all image files from input directory
    print(f"Scanning directory: {args.input_dir}")
    image_files = get_image_files(args.input_dir)
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} image files:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16
    vae_scale = 0.18215
    image_size = 512
    latent_size = image_size // 8
    
    print(f"\nUsing device: {device}")
    print(f"Fixed parameters - Steps: {steps}, CFG Scale: {cfg_scale}")
    print(f"Processing {len(image_files)} images")
    
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
    
    # Prepare text embedding once
    print("Encoding text prompt...")
    caption_embs, emb_masks = t5_model.get_text_embeddings([args.prompt])
    caption_embs = caption_embs[:, None]
    null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]
    
    # Process each image file
    for idx, image_path in enumerate(image_files):
        print(f"\nProcessing image {idx+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Load and process condition image  
        condition_img = Image.open(image_path).convert('RGB')
        condition_img = condition_img.resize((image_size, image_size), Image.LANCZOS)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        condition_tensor = transform(condition_img).unsqueeze(0).to(device).to(torch.float32)
        
        # Encode condition image to latent space
        print("  Encoding condition image to latent space...")
        with torch.no_grad():
            posterior = vae.encode(condition_tensor).latent_dist
            condition_latent = posterior.sample() * vae_scale
        
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
        
        # Convert to PIL image
        generated_img = transforms.ToPILImage()(samples[0].cpu().clamp(-1, 1) * 0.5 + 0.5)
        
        # Create simple comparison image
        comparison_img = create_simple_comparison(condition_img, generated_img)
        
        # Save comparison image with numbered filename
        comparison_save_path = os.path.join(args.output_dir, f'comparison_{idx:03d}.png')
        comparison_img.save(comparison_save_path)
        
        print(f"  Saved: {comparison_save_path}")
    
    print(f"\n=== Results ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameters used: Steps={steps}, CFG Scale={cfg_scale}")
    print(f"Processed {len(image_files)} images")
    print(f"Generated comparison files: comparison_000.png ~ comparison_{len(image_files)-1:03d}.png")

if __name__ == "__main__":
    main() 