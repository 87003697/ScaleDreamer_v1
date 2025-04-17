# CUDA_VISIBLE_DEVICES=0 python launch.py \
#     --config configs/next_0/DF415_debug_v0.yaml  \
#     --train \
#     data.prompt_library="dreamfusion_415_prompt_library"

CUDA_VISIBLE_DEVICES=0 python launch.py \
    --config configs/next_0/DF415_debug_v0.yaml  \
    --train \
    data.prompt_library="dreamfusion_415_prompt_library"
