# Train on 1024px
python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 train_scripts/train_controlnet.py configs/pixart_app_config/PixArt_xl2_img1024_controlHed.py --work-dir output/pixartcontrolnet-xl2-img1024

# Train on 512px
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 train_scripts/train_controlnet.py configs/pixart_app_config/PixArt_xl2_img512_controlHed.py --work-dir output/pixartcontrolnet-xl2-img512