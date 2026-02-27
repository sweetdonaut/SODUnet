python trainer.py \
    --bs 16 \
    --lr 0.001 \
    --epochs 30 \
    --gpu_id 0 \
    --checkpoint_path ../checkpoints/4channel \
    --patch_size 128 \
    --training_dataset_path ../data/grid_stripe_4channel/train/good/ \
    --img_format tiff \
    --num_defects_range 4 10 \
    --cache_size 100 \
    --defect_mode gaussian

# PSF mode example:
# python trainer.py \
#     --bs 16 --lr 0.001 --epochs 30 --gpu_id 0 \
#     --checkpoint_path ../checkpoints/4channel \
#     --patch_size 128 \
#     --training_dataset_path ../data/grid_stripe_4channel/train/good/ \
#     --img_format tiff --num_defects_range 4 10 --cache_size 100 \
#     --defect_mode psf --psf_type type1 type2
