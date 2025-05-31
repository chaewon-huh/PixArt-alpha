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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='output/pixartcontrolnet-xl2-img512/checkpoints/epoch_8_step_2000.pth')
    parser.add_argument('--condition_images', nargs='+', default=['data/images/0.png', 'data/images/1.png', 'data/images/2.png'])
    parser.add_argument('--prompt', type=str, default='A high-quality, photorealistic image of a healthy, anatomically normal human baby, reconstructed solely from the fetal shape and pose of the input 3D ultrasound image of an fetal.')
    parser.add_argument('--output_dir', type=str, default='test_outputs')
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg_scale', type=float, default=4.5)
    parser.add_argument('--steps', type=int, default=20)
    
    args = parser.parse_args()
    
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16
    vae_scale = 0.18215
    image_size = 512
    latent_size = image_size // 8
    
    print(f"Using device: {device}")
    
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
        
        # Generate samples
        print(f"Generating {args.num_samples} samples...")
        for sample_idx in range(args.num_samples):
            torch.manual_seed(args.seed + sample_idx)
            
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
                cfg_scale=args.cfg_scale,
                model_kwargs=model_kwargs
            )
            
            print(f"  Sampling step {sample_idx+1}/{args.num_samples}...")
            with torch.no_grad():
                samples = dpm_solver.sample(
                    z,
                    steps=args.steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            
            # Decode
            samples = vae.decode(samples.to(torch.float32) / vae_scale).sample
            
            # Save image
            output_path = os.path.join(args.output_dir, f'condition{idx}_sample{sample_idx}.png')
            save_image = transforms.ToPILImage()(samples[0].cpu().clamp(-1, 1) * 0.5 + 0.5)
            save_image.save(output_path)
            print(f"  Saved: {output_path}")
            
            # Also save condition visualization  
            if sample_idx == 0:
                # Save original condition image
                condition_orig_path = os.path.join(args.output_dir, f'condition{idx}_original.png')
                condition_img.save(condition_orig_path)
                print(f"  Saved original: {condition_orig_path}")
                
                # Save VAE reconstruction of condition
                condition_recon = vae.decode(condition_latent.to(torch.float32) / vae_scale).sample
                condition_recon_path = os.path.join(args.output_dir, f'condition{idx}_vae_recon.png')
                condition_recon_img = transforms.ToPILImage()(condition_recon[0].cpu().clamp(-1, 1) * 0.5 + 0.5)
                condition_recon_img.save(condition_recon_path)
                print(f"  Saved VAE reconstruction: {condition_recon_path}")
    
    print(f"\nAll results saved to {args.output_dir}")
    print("\nSummary:")
    print(f"- Processed {len(args.condition_images)} condition images")
    print(f"- Generated {args.num_samples} samples per condition")
    print(f"- Total {len(args.condition_images) * args.num_samples} images generated")

if __name__ == "__main__":
    main() 