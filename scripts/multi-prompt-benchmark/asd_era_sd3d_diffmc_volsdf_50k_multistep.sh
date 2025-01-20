# CUDA_VISIBLE_DEVICES=7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_diffmc_volsdf_50k_multistep.yaml \
#     --train \
#     data.prompt_library="dreamfusion_415_prompt_library"

# CUDA_VISIBLE_DEVICES=7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_diffmc_volsdf_50k_multistep.yaml \
#     --train \
#      data.image_library="sdxl_3d_animation_v1_7_image_library" \
#      data.image_root_dir="datasets/sdxl_3d_animation_v1_7"


CUDA_VISIBLE_DEVICES=5,6  python launch.py \
    --config configs/multi-prompt_benchmark/asd_era_sd3d_diffmc_volsdf_50k_multistep.yaml \
    --train \
     data.image_library="sdxl_3d_animation_v1_7_image_library" \
     data.image_root_dir="datasets/sdxl_3d_animation_v1_7"

# CUDA_VISIBLE_DEVICES=7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_era_sd3d_diffmc_volsdf_50k_multistep.yaml \
#     --test \
#      data.image_library="sdxl_3d_animation_v1_7_image_library" \
#      data.image_root_dir="datasets/sdxl_3d_animation_v1_7"