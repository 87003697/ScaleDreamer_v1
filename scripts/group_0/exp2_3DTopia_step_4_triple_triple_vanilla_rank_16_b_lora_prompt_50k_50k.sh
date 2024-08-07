CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python launch.py \
    --config configs/group_0/3DTopia_step_4_triple_triple_vanilla_rank_16_b_lora_prompt_50k_50k.yaml \
    --train \
    data.prompt_library="3DTopia_361k_prompt_library" \
    data.prompt_processor.cache_dir=.threestudio_cache/text_embeddings_3DTopia_361k 

# CUDA_VISIBLE_DEVICES=0  python launch.py \
#     --config configs/group_0/3DTopia_step_4_triple_triple_vanilla_rank_16_b_lora_prompt_50k_50k.yaml \
#     --train \
#     data.prompt_library="3DTopia_361k_prompt_library" \
#     data.prompt_processor.cache_dir=.threestudio_cache/text_embeddings_3DTopia_361k 