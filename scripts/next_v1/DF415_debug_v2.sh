CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch.py \
    --config configs/next_0/DF415_debug_v2.yaml  \
    --train \
    data.prompt_library="dreamfusion_415_prompt_library"

# CUDA_VISIBLE_DEVICES=0 python launch.py \
#     --config configs/next_0/DF415_debug_v2.yaml  \
#     --train \
#     data.prompt_library="dreamfusion_415_prompt_library"
