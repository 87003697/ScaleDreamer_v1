# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  python launch.py \
#     --config configs/multi-prompt_benchmark/asd_mv+sd_sd3d_v3_dmtet_volsdf_10k.yaml \
#     --train \
#     system.prompt_processor.prompt_library="dreamfusion_415_prompt_library"

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0  python launch.py \
    --config configs/multi-prompt_benchmark/asd_mv+sd_sd3d_v3_dmtet_volsdf_10k.yaml \
    --train \
    system.prompt_processor.prompt_library="magic3d_15_prompt_library"
