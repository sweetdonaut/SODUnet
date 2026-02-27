python inference.py \
    --model_path ../checkpoints/4channel/BgRemoval_lr0.001_ep30_bs16_128x128.pth \
    --test_path ../data/grid_stripe_4channel/test/ \
    --output_dir ../output/pytorch \
    --img_format tiff
