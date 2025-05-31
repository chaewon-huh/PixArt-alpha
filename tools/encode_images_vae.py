import sys
from pathlib import Path
import os
import json
import torch
import numpy as np
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
import argparse

# 프로젝트 루트 경로 추가 (hed.py와 동일한 방식)
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

class VAEEncodeDataset(Dataset):
    def __init__(self, root_data_dir, data_info_json_path, condition_source_image_dir, image_resize=512):
        """
        Args:
            root_data_dir (str): The root directory where data_info.json is and where features will be saved (e.g. ./data).
            data_info_json_path (str): Full path to the data_info.json file.
            condition_source_image_dir (str): Directory containing the original condition images.
            image_resize (int): Image resize dimension before VAE encoding.
        """
        with open(data_info_json_path, 'r') as f:
            self.data_info_list = json.load(f)
        
        self.condition_source_image_dir = condition_source_image_dir
        self.root_data_dir = root_data_dir
        
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(image_resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_resize),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]), # VAE expects inputs in [-1, 1]
        ])

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):
        item_info = self.data_info_list[idx]
        # 'path' in data_info.json is like 'images/0.png'
        # We need the filename (0.png) to build the path in condition_source_image_dir
        filename_from_data_info = Path(item_info['path']).name 
        
        source_image_path = os.path.join(self.condition_source_image_dir, filename_from_data_info)

        try:
            image = Image.open(source_image_path)
            transformed_image = self.transform(image)
            # This save_path_suffix will be used to construct the final NPZ path
            # e.g., if item_info['path'] is 'images/0.png', suffix is 'images/0.npz'
            save_path_suffix = item_info['path'].replace('.png', '.npz') 
            return transformed_image, save_path_suffix
        except Exception as e:
            print(f"Error loading or transforming image {source_image_path}: {e}")
            # Return None to allow skipping in the DataLoader loop if an image is corrupted
            return None, None 

def main(args):
    # Ensure data_info.json path is constructed correctly if a relative path is given
    if not os.path.isabs(args.data_info_json):
        data_info_json_full_path = os.path.join(args.output_root_dir, "partition", "data_info.json") 
    else:
        data_info_json_full_path = args.data_info_json
    if not os.path.exists(data_info_json_full_path):
        print(f"Error: data_info.json not found at {data_info_json_full_path}")
        return

    dataset = VAEEncodeDataset(
        root_data_dir=args.output_root_dir, 
        data_info_json_path=data_info_json_full_path,
        condition_source_image_dir=args.condition_source_dir,
        image_resize=args.image_size
    )
    
    # Filter out None items from dataset (e.g. due to image loading errors)
    # This is a simple way; for large datasets, consider modifying __getitem__ or using a custom collate_fn
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                            collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x[0] is not None]))

    if len(dataloader) == 0:
        print("No valid images found to process. Exiting.")
        return

    accelerator = Accelerator(mixed_precision='fp16' if args.fp16 else 'no')
    
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_model_path)
    except Exception as e:
        print(f"Error loading VAE model from {args.vae_model_path}: {e}")
        return
        
    vae.requires_grad_(False)
    vae.eval()
    
    vae = accelerator.prepare(vae)
    dataloader = accelerator.prepare(dataloader)

    # Output directory for the NPZ files, e.g., ./data/condition_feature_512/
    condition_feature_dir_name = f"condition_feature_{args.image_size}"
    full_output_dir_base = os.path.join(args.output_root_dir, condition_feature_dir_name)

    for batch_data in tqdm(dataloader):
        if not batch_data: # Skip if batch is empty after filtering Nones
            continue
        batch_images, batch_save_path_suffixes = batch_data
        if batch_images is None or len(batch_images) == 0: # Double check after collate_fn
            continue
            
        with torch.no_grad():
            posterior = vae.encode(batch_images).latent_dist
            encoded_features = torch.cat([posterior.mean, posterior.std], dim=1).cpu().numpy()

        for i in range(encoded_features.shape[0]):
            single_feature = encoded_features[i] 
            save_path_suffix = batch_save_path_suffixes[i] 
            
            # Final save path, e.g., ./data/condition_feature_512/images/0.npz
            full_save_path = os.path.join(full_output_dir_base, save_path_suffix)
            
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            np.savez_compressed(full_save_path, arr_0=single_feature) # Save with key 'arr_0' for compatibility

    print(f"VAE encoded features saved in {full_output_dir_base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode condition images using VAE and save as NPZ.")
    parser.add_argument("--data_info_json", type=str, default="partition/data_info.json",
                        help="Path to the data_info.json file, relative to output_root_dir if not absolute.")
    parser.add_argument("--condition_source_dir", type=str, required=True,
                        help="Directory containing the original condition images (e.g., ./data_raw/MyConditionImages).")
    parser.add_argument("--output_root_dir", type=str, default="./data",
                        help="Root directory for data_info.json and to save encoded NPZ files (e.g., ./data). Output will be in output_root_dir/condition_feature_SIZE/")
    parser.add_argument("--vae_model_path", type=str, default="output/pretrained_models/sd-vae-ft-ema",
                        help="Path to the pretrained VAE model.")
    parser.add_argument("--image_size", type=int, default=512, choices=[256, 512, 1024],
                        help="Image size for VAE processing.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (fp16).")
    
    args = parser.parse_args()
    main(args) 