CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  python launch.py \
    --config configs/multi-prompt_benchmark/asd_mv+sd_sd3d_v1_dmtet_volsdf_500k_multistep.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.prompt_processor.cache_dir=.threestudio_cache/text_embeddings_3DTopia_361k 

# CUDA_VISIBLE_DEVICES=0  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_mv+sd_sd3d_v1_dmtet_volsdf_50k_multistep.yaml \
#     --train \
#     data.prompt_library="magic3d_15_prompt_library" \


# CUDA_VISIBLE_DEVICES=0  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_mv+sd_sd3d_v1_dmtet_volsdf_500k_multistep.yaml \
#     --test \
#     data.prompt_library="3DTopia_361k_prompt_library" \
#     data.prompt_processor.cache_dir=.threestudio_cache/text_embeddings_3DTopia_361k \
#     system.weights="outputs/asd_mv+sd_sd3d_v1_dmtet_volsdf_500k_multistep/3DTopia_361k_prompt_library/ckpts/epoch=0-step=20000.ckpt"