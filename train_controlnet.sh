python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 \
  train_scripts/train_controlnet.py \
  configs/pixart_app_config/PixArt_xl2_img512_controlHed.py \
  --work-dir output/pixartcontrolnet-xl2-img512 \
  --resume_from /workspace/PixArt-alpha/output/pixartcontrolnet-xl2-img512/checkpoints/epoch_148_step_37000.pth \
  --resume_optimizer \
  --resume_lr_scheduler 